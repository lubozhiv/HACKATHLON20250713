import json
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
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
    Service for embedding documents using BERT and storing them in Qdrant vector database.
    """

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333,
                 collection_name: str = "reddit_pain_documents"):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence transformer model to use
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
        """
        self.model_name = model_name
        self.collection_name = collection_name

        # Initialize BERT model
        self.model = SentenceTransformer(model_name)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port)

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_collection(self, recreate: bool = False) -> None:
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

    def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of embedding vectors
        """
        logger.info(f"Generating embeddings for {len(documents)} documents")

        texts = [doc.page_content for doc in documents]
        embeddings = []
        batch_size = 32

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            embeddings.extend(batch_embeddings)

        return embeddings

    def prepare_points(self, documents: List[Document], embeddings: List[np.ndarray]) -> List[PointStruct]:
        """
        Prepare points for Qdrant insertion.

        Args:
            documents: List of LangChain Document objects
            embeddings: List of embedding vectors

        Returns:
            List of PointStruct objects for Qdrant
        """
        points = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            point_id, content_hash = self._generate_document_id_and_hash(doc)

            metadata = doc.metadata.copy()
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata[key] = float(value)
                elif value is None:
                    metadata[key] = ""

            metadata['text'] = doc.page_content
            metadata['text_length'] = len(doc.page_content)
            metadata['content_hash'] = content_hash  # Store the hash in payload

            point = PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            )
            points.append(point)

        return points

    def insert_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        """
        Insert documents into Qdrant database, avoiding re-embedding and re-uploading
        of unchanged documents.

        Args:
            documents: List of LangChain Document objects
            batch_size: Number of documents to process and insert at once
        """
        documents_to_process: List[Document] = []
        point_ids_to_process: List[str] = []
        original_indices_to_process: List[int] = []  # To map back to original documents_to_process order

        logger.info(f"Checking {len(documents)} documents for changes before processing...")

        # First pass: Determine which documents need processing
        all_incoming_point_ids = []
        incoming_point_id_map = {}  # Map point_id to its original Document object and calculated hash
        for idx, doc in enumerate(documents):
            point_id, current_hash = self._generate_document_id_and_hash(doc)
            all_incoming_point_ids.append(point_id)
            incoming_point_id_map[point_id] = {'doc': doc, 'hash': current_hash, 'original_idx': idx}

        # Retrieve existing hashes for all potential point_ids in one go
        # This is more efficient than individual lookups inside the loop
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

        # Generate embeddings only for the documents that need processing
        embeddings = self.embed_documents(documents_to_process)

        # Prepare points with their now-known stable IDs and metadata
        # We need to ensure the order of documents_to_process matches embeddings
        # For simplicity, we re-run prepare_points, but if you want to optimize further,
        # you could pass original_indices_to_process and filter from the original list.
        # However, passing `documents_to_process` and `embeddings` (which match in order) is cleaner.
        points_to_upsert = []
        for doc_to_embed, embedding in zip(documents_to_process, embeddings):
            point_id, content_hash = self._generate_document_id_and_hash(
                doc_to_embed)  # Re-derive for point_id and hash

            metadata = doc_to_embed.metadata.copy()
            for key, value in metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    metadata[key] = float(value)
                elif value is None:
                    metadata[key] = ""
            metadata['text'] = doc_to_embed.page_content
            metadata['text_length'] = len(doc_to_embed.page_content)
            metadata['content_hash'] = content_hash  # Store the hash in payload

            points_to_upsert.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=metadata
            ))

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

    def search_similar_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query text
            limit: Number of results to return

        Returns:
            List of similar documents with metadata
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]

        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=limit
        )

        # Format results
        results = []
        for result in search_results:
            results.append({
                'id': str(result.id),  # Ensure ID is string
                'score': result.score,
                'metadata': result.payload
            })

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        Returns:
            Dictionary containing collection information
        """
        try:
            info: CollectionInfo = self.qdrant_client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': info.result.vectors_count,
                'indexed_vectors_count': info.result.indexed_vectors_count,
                'points_count': info.result.points_count
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
