import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList, CollectionInfo
from langchain.schema import Document
import uuid
import hashlib
from tqdm import tqdm
import logging

# Set up logging
logger = logging.getLogger(__name__)


class QdrantEmbeddingService:
    """
    Service for embedding documents using OpenAI and storing them in Qdrant vector database.
    Enhanced to work with the improved noise filtering system.
    """

    def __init__(self,
                 model_name: str = "text-embedding-ada-002",
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "reddit_pain_documents"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the OpenAI embedding model to use
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
        """
        self.model_name = model_name
        self.collection_name = collection_name

        # Initialize OpenAI embeddings model
        self.model = OpenAIEmbeddings(
            model=model_name,
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            show_progress_bar=True
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port)

        # OpenAI's text-embedding-ada-002 has 1536 dimensions
        self.embedding_dim = 1536

    def create_collection(self, recreate: bool = True) -> None:
        """
        Create Qdrant collection if it doesn't exist.

        Args:
            recreate: If True, delete existing collection and create new one
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if collection_exists and recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.qdrant_client.delete_collection(self.collection_name)
                collection_exists = False

            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    def _generate_document_id_and_hash(self, doc: Document) -> Tuple[str, str]:
        """
        Generates a consistent ID and content hash for a document.
        """
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        source_info = doc.metadata.get('source_file', '')
        consistent_id_string = f"{content_hash}-{source_info}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, consistent_id_string))
        return point_id, content_hash

    def _get_existing_point_info(self, point_ids: List[str]) -> Dict[str, str]:
        """
        Retrieves the content hash for existing points from Qdrant.
        Returns a dictionary mapping point_id to its stored content_hash.
        """
        if not point_ids:
            return {}

        existing_hashes = {}
        try:
            # Fetch only the 'content_hash' field
            retrieved_points = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=False
            )
            for point in retrieved_points:
                if point.payload and 'content_hash' in point.payload:
                    existing_hashes[str(point.id)] = point.payload['content_hash']
        except Exception as e:
            logger.warning(f"Could not retrieve existing point info for IDs {point_ids}: {e}")
        return existing_hashes

    def embed_documents(self, documents: List[Document], batch_size: int = 50) -> List[np.ndarray]:
        """
        Generate embeddings for a list of documents with token limit handling.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process at once (adjust based on content length)

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")

        all_embeddings = []
        texts = [doc.page_content for doc in documents]

        # Process in batches to handle token limits
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]

            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            batch_token_count = sum(len(text) for text in batch_texts) // 4

            # If batch is too large, reduce batch size dynamically
            if batch_token_count > 250000:  # Leave some margin under 300k limit
                # Reduce batch size and retry
                smaller_batch_size = max(1, batch_size // 2)
                logger.warning(
                    f"Batch too large ({batch_token_count} tokens), reducing batch size to {smaller_batch_size}")

                # Process this batch with smaller size
                for j in range(i, min(i + batch_size, len(texts)), smaller_batch_size):
                    small_batch = texts[j:j + smaller_batch_size]
                    try:
                        batch_embeddings = self.model.embed_documents(small_batch)
                        all_embeddings.extend([np.array(emb) for emb in batch_embeddings])
                    except Exception as e:
                        logger.error(f"Error processing small batch starting at index {j}: {str(e)}")
                        # If even a small batch fails, process documents individually
                        for text in small_batch:
                            try:
                                single_embedding = self.model.embed_query(text)
                                all_embeddings.append(np.array(single_embedding))
                            except Exception as single_error:
                                logger.error(f"Failed to embed single document: {str(single_error)}")
                                # Create a zero vector as fallback
                                all_embeddings.append(np.zeros(self.embedding_dim))
            else:
                # Normal batch processing
                try:
                    batch_embeddings = self.model.embed_documents(batch_texts)
                    all_embeddings.extend([np.array(emb) for emb in batch_embeddings])
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                    # Fallback to individual processing for this batch
                    for text in batch_texts:
                        try:
                            single_embedding = self.model.embed_query(text)
                            all_embeddings.append(np.array(single_embedding))
                        except Exception as single_error:
                            logger.error(f"Failed to embed single document: {str(single_error)}")
                            # Create a zero vector as fallback
                            all_embeddings.append(np.zeros(self.embedding_dim))

        logger.info(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings

    def _process_metadata_for_qdrant(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata to ensure compatibility with Qdrant.
        Convert numpy types, handle nested structures, and add filtering stats.
        """
        processed_metadata = {}

        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                processed_metadata[key] = float(value)
            elif isinstance(value, dict):
                # Handle nested dictionaries (like filtering_stats)
                if key == 'filtering_stats':
                    # Flatten filtering stats for easier searching
                    processed_metadata['original_comment_count'] = value.get('original_count', 0)
                    processed_metadata['filtered_comment_count'] = value.get('filtered_count', 0)
                    processed_metadata['comment_removal_rate'] = value.get('removal_rate', 0.0)
                    processed_metadata['adaptive_threshold'] = value.get('adaptive_threshold', 0.0)

                    # Score distribution stats
                    score_dist = value.get('score_distribution', {})
                    processed_metadata['score_min'] = score_dist.get('min', 0.0)
                    processed_metadata['score_max'] = score_dist.get('max', 0.0)
                    processed_metadata['score_mean'] = score_dist.get('mean', 0.0)
                    processed_metadata['score_std'] = score_dist.get('std', 0.0)
                else:
                    # For other nested dicts, convert to JSON string or flatten
                    try:
                        processed_metadata[key] = json.dumps(value) if value else ""
                    except (TypeError, ValueError):
                        processed_metadata[key] = str(value)
            elif isinstance(value, list):
                # Convert lists to JSON strings for storage
                try:
                    processed_metadata[key] = json.dumps(value) if value else "[]"
                except (TypeError, ValueError):
                    processed_metadata[key] = str(value)
            elif value is None:
                processed_metadata[key] = ""
            else:
                processed_metadata[key] = value

        return processed_metadata

    def prepare_points(self, documents: List[Document], embeddings: List[np.ndarray]) -> List[PointStruct]:
        """
        Prepare points for Qdrant insertion with enhanced metadata processing.

        Args:
            documents: List of LangChain Document objects
            embeddings: List of embedding vectors

        Returns:
            List of PointStruct objects for Qdrant
        """
        points = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id, content_hash = self._generate_document_id_and_hash(doc)

            # Process metadata to handle the enhanced filtering stats
            processed_metadata = self._process_metadata_for_qdrant(doc.metadata)

            # Add document text and basic stats
            processed_metadata['text'] = doc.page_content
            processed_metadata['text_length'] = len(doc.page_content)
            processed_metadata['content_hash'] = content_hash

            # Add filtering quality indicators
            if 'comment_similarity_score' in doc.metadata:
                processed_metadata['comment_quality_score'] = float(doc.metadata['comment_similarity_score'])

            # Calculate text quality metrics
            processed_metadata['word_count'] = len(doc.page_content.split())
            processed_metadata['has_technical_content'] = any(
                term in doc.page_content.lower()
                for term in ['api', 'code', 'function', 'bug', 'error', 'github', 'python', 'javascript']
            )

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=processed_metadata
            )
            points.append(point)

        return points

    def insert_documents(self, documents: List[Document], batch_size: int = 10, embedding_batch_size: int = 20) -> None:
        """
        Insert documents into Qdrant database, avoiding re-embedding and re-uploading
        of unchanged documents. Enhanced to work with improved filtering metadata.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to insert into Qdrant at once
            embedding_batch_size: Number of documents to embed at once (smaller to handle token limits)
        """
        documents_to_process: List[Document] = []
        point_ids_to_process: List[str] = []
        original_indices_to_process: List[int] = []

        logger.info(f"Checking {len(documents)} documents for changes before processing...")

        # First pass: Determine which documents need processing
        all_incoming_point_ids = []
        incoming_point_id_map = {}
        for idx, doc in enumerate(documents):
            point_id, current_hash = self._generate_document_id_and_hash(doc)
            all_incoming_point_ids.append(point_id)
            incoming_point_id_map[point_id] = {'doc': doc, 'hash': current_hash, 'original_idx': idx}

        # Retrieve existing hashes for all potential point_ids in one go
        existing_point_hashes = self._get_existing_point_info(all_incoming_point_ids)

        for point_id, info in incoming_point_id_map.items():
            doc = info['doc']
            current_hash = info['hash']
            original_idx = info['original_idx']

            stored_hash = existing_point_hashes.get(point_id)

            if stored_hash is None:
                # Point does not exist, needs to be inserted
                documents_to_process.append(doc)
                point_ids_to_process.append(point_id)
                original_indices_to_process.append(original_idx)
            elif stored_hash != current_hash:
                # Point exists but content has changed, needs to be updated
                documents_to_process.append(doc)
                point_ids_to_process.append(point_id)
                original_indices_to_process.append(original_idx)
            else:
                # Point exists and content is the same, skip
                logger.debug(f"Skipping document with ID {point_id} (hash {current_hash}): already up-to-date.")

        if not documents_to_process:
            logger.info("All documents are already up-to-date in Qdrant. No new insertions/updates needed.")
            return

        logger.info(f"Found {len(documents_to_process)} documents requiring embedding/upserting.")

        # Log filtering statistics
        total_original_comments = sum(
            doc.metadata.get('filtering_stats', {}).get('original_count', 0)
            for doc in documents_to_process
        )
        total_filtered_comments = sum(
            doc.metadata.get('filtering_stats', {}).get('filtered_count', 0)
            for doc in documents_to_process
        )

        if total_original_comments > 0:
            overall_removal_rate = 1.0 - (total_filtered_comments / total_original_comments)
            logger.info(f"Overall filtering stats: {total_original_comments} -> {total_filtered_comments} comments "
                        f"(removal rate: {overall_removal_rate:.2%})")

        # Generate embeddings with smaller batch size to handle token limits
        embeddings = self.embed_documents(documents_to_process, batch_size=embedding_batch_size)

        # Prepare points with enhanced metadata
        points_to_upsert = self.prepare_points(documents_to_process, embeddings)

        # Insert in batches
        for i in tqdm(range(0, len(points_to_upsert), batch_size), desc="Upserting into Qdrant"):
            batch_points = points_to_upsert[i:i + batch_size]

            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points
                )
            except Exception as e:
                logger.error(f"Error upserting batch {i // batch_size + 1}: {str(e)}")
                raise

        logger.info(f"Successfully upserted {len(documents_to_process)} documents (new or updated).")

    def search_similar_documents(self,
                                 query: str,
                                 limit: int = 10,
                                 filter_conditions: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database with enhanced filtering options.

        Args:
            query: Search query text
            limit: Number of results to return
            filter_conditions: Optional filter conditions for metadata

        Returns:
            List of similar documents with metadata
        """
        # Generate query embedding
        query_embedding = self.model.embed_query(query)
        query_embedding = np.array(query_embedding)

        # Search in Qdrant with optional filtering
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding.tolist(),
            "limit": limit,
            "with_payload": True,
            "with_vectors": False
        }

        if filter_conditions:
            search_params["query_filter"] = filter_conditions

        search_results = self.qdrant_client.search(**search_params)

        # Format results with enhanced metadata
        results = []
        for result in search_results:
            payload = result.payload

            # Extract filtering quality metrics
            quality_info = {
                'comment_quality_score': payload.get('comment_quality_score', 0.0),
                'original_comment_count': payload.get('original_comment_count', 0),
                'filtered_comment_count': payload.get('filtered_comment_count', 0),
                'comment_removal_rate': payload.get('comment_removal_rate', 0.0),
                'has_technical_content': payload.get('has_technical_content', False),
                'adaptive_threshold': payload.get('adaptive_threshold', 0.0)
            }

            results.append({
                'id': str(result.id),
                'score': result.score,
                'metadata': payload,
                'quality_info': quality_info
            })

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection with enhanced statistics.

        Returns:
            Dictionary containing collection information
        """
        try:
            info: CollectionInfo = self.qdrant_client.get_collection(self.collection_name)

            # Get sample of documents to calculate quality stats
            sample_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )

            # Calculate quality statistics
            quality_stats = {
                'total_documents': 0,
                'avg_comment_quality': 0.0,
                'avg_removal_rate': 0.0,
                'technical_content_ratio': 0.0
            }

            if sample_results[0]:  # If we have documents
                documents = sample_results[0]
                quality_stats['total_documents'] = len(documents)

                quality_scores = [doc.payload.get('comment_quality_score', 0.0) for doc in documents]
                removal_rates = [doc.payload.get('comment_removal_rate', 0.0) for doc in documents]
                technical_flags = [doc.payload.get('has_technical_content', False) for doc in documents]

                if quality_scores:
                    quality_stats['avg_comment_quality'] = sum(quality_scores) / len(quality_scores)
                if removal_rates:
                    quality_stats['avg_removal_rate'] = sum(removal_rates) / len(removal_rates)
                if technical_flags:
                    quality_stats['technical_content_ratio'] = sum(technical_flags) / len(technical_flags)

            return {
                'name': self.collection_name,
                'vectors_count': info.result.vectors_count,
                'indexed_vectors_count': info.result.indexed_vectors_count,
                'points_count': info.result.points_count,
                'quality_stats': quality_stats
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}