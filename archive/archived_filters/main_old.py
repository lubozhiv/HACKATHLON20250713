import json
import os
from typing import List, Dict, Any
from langchain.schema import Document
from AdvancedPreprocessingImplementation.no_1_filter_1_quality_metrics import filter_by_quality
from AdvancedPreprocessingImplementation.no_2_filter_2_pain_detection import AdvancedPainDetector
from archive.archived_filters.no_4_filter_3_noise import NoiseDetector
import numpy as np
from AdvancedPreprocessingImplementation.no_4_embedded_to_qdrant import QdrantEmbeddingService
from qdrant_search_idea_generator import QdrantSearchAndIdeaGenerator




def json_to_langchain_documents(json_data: List[Dict[str, Any]], filename: str = "") -> List[Document]:
    """
    Convert JSON topics with comments to LangChain Document objects.
    """
    documents = []

    for topic in json_data:
        # Extract title and selftext
        title = topic.get("title", "")
        selftext = topic.get("selftext", "")

        # Extract comment bodies
        comments = topic.get("comments", [])
        comment_bodies = []

        for comment in comments:
            body = comment.get("body", "")
            if body:
                comment_bodies.append(body)

        # Create page_content by combining title, selftext, and comments
        page_content_parts = []

        if title:
            page_content_parts.append(f"Title: {title}")

        if selftext:
            page_content_parts.append(f"Content: {selftext}")

        if comment_bodies:
            page_content_parts.append("Comments:")
            for i, comment_body in enumerate(comment_bodies, 1):
                page_content_parts.append(f"Comment {i}: {comment_body}")

        page_content = "\n\n".join(page_content_parts)

        # Create metadata with filterable attributes
        metadata = {
            "id": topic.get("id", ""),
            "subreddit": topic.get("subreddit", ""),
            "score": topic.get("score", 0),
            "upvote_ratio": topic.get("upvote_ratio", 0.0),
            "num_comments": topic.get("num_comments", 0),
            "created_utc": topic.get("created_utc", 0),
            "author": topic.get("author", ""),
            "url": topic.get("url", ""),
            "permalink": topic.get("permalink", ""),
            "is_self": topic.get("is_self", False),
            "distinguished": topic.get("distinguished"),
            "stickied": topic.get("stickied", False),
            "over_18": topic.get("over_18", False),
            "spoiler": topic.get("spoiler", False),
            "locked": topic.get("locked", False),
            "comments_extracted": topic.get("comments_extracted", 0),
            "source_file": filename
        }

        # Create Document object
        document = Document(
            page_content=page_content,
            metadata=metadata
        )

        documents.append(document)

    return documents


def load_json_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON data must be a dictionary or list of dictionaries")


def load_all_json_files_from_directory(directory: str) -> List[Document]:
    """
    Load all JSON files from a directory and convert them to LangChain Documents.
    """
    all_documents = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                json_data = load_json_from_file(file_path)
                documents = json_to_langchain_documents(json_data, filename)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    return all_documents


def create_document_id_mapping(documents: List[Document]) -> Dict[str, Document]:
    """
    Create a mapping from document IDs to Document objects for efficient lookup.
    """
    # Create mapping using the same ID generation logic as QdrantEmbeddingService
    import hashlib
    import uuid

    id_to_doc = {}
    for doc in documents:
        # Use the same logic as in QdrantEmbeddingService._generate_document_id_and_hash
        content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
        source_info = doc.metadata.get('source_file', '')
        consistent_id_string = f"{content_hash}-{source_info}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, consistent_id_string))
        id_to_doc[point_id] = doc

    return id_to_doc


def sync_metadata_with_qdrant(embedding_service: QdrantEmbeddingService,
                              documents: List[Document]) -> None:
    """
    Synchronize all metadata from documents to Qdrant.
    This ensures that quality and pain metrics are properly stored in Qdrant.
    """
    print("Synchronizing metadata with Qdrant...")

    # Create document ID mapping
    id_to_doc = create_document_id_mapping(documents)

    # Update metadata in batches
    batch_size = 100
    doc_ids = list(id_to_doc.keys())

    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i + batch_size]

        for doc_id in batch_ids:
            doc = id_to_doc[doc_id]

            # Prepare payload with all metadata
            payload = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.integer, np.floating)):
                    payload[key] = float(value)
                elif value is None:
                    payload[key] = ""
                else:
                    payload[key] = value

            # Add text content
            payload['text'] = doc.page_content
            payload['text_length'] = len(doc.page_content)

            try:
                embedding_service.qdrant_client.set_payload(
                    collection_name=embedding_service.collection_name,
                    payload=payload,
                    points=[doc_id]
                )
            except Exception as e:
                print(f"Error updating metadata for document {doc_id}: {e}")
                continue

    print(f"Metadata synchronization complete for {len(documents)} documents")


def interactive_search(embedding_service: QdrantEmbeddingService):
    """
    Handle user interaction and display results
    """
    idea_generator = QdrantSearchAndIdeaGenerator(embedding_service)

    print("\nProduct Idea Generator - Enter a query or 'exit' to quit")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == 'exit':
            break

        if not query:
            print("Please enter a valid query.")
            continue

        print(f"\nProcessing query: '{query}'...")

        try:
            # Process the query through the generator
            results = idea_generator.process_query(query)

            # Display search results
            if results["search_results"]:
                print(f"\nTop {len(results['search_results'])} relevant discussions:")
                for i, result in enumerate(results["search_results"], 1):
                    print(f"{i}. Score: {result['score']:.2f} | Quality: {result['quality_score']:.2f} | "
                          f"Pain: {result['pain_score']:.2f} | Noise: {result['is_noise']}")
                    print(f"   Source: {result['payload'].get('subreddit', 'N/A')}")
                    print(f"   Content preview: {result['payload']['text'][:200]}...\n")
            else:
                print("No search results found.")

            # Display generated ideas
            print("\nGenerated Product Ideas:\n")
            print(results["ideas"])

        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Load all documents from the datasets directory
    print("Loading documents from datasets directory...")
    all_documents = load_all_json_files_from_directory('datasets')
    print(f"Total documents loaded: {len(all_documents)}")

    # Apply quality filter
    print("Applying quality filter...")
    quality_threshold = 0.5
    quality_filtered_documents = filter_by_quality(all_documents, min_score=quality_threshold)
    print(f"Documents after quality filtering: {len(quality_filtered_documents)}")

    # Apply pain detection filter
    print("Applying pain detection filter...")
    pain_detector = AdvancedPainDetector()
    pain_filtered_documents = pain_detector.filter_documents_by_pain(
        quality_filtered_documents,
        min_pain_score=0.005,
        max_pain_score=1.0
    )
    print(f"Documents after pain filtering: {len(pain_filtered_documents)}")

    # Insert documents into Qdrant with embeddings
    print("Inserting documents into Qdrant...")
    embedding_service = QdrantEmbeddingService()
    embedding_service.create_collection(recreate=False)
    embedding_service.insert_documents(pain_filtered_documents)

    # Sync all metadata to Qdrant (ensures quality and pain metrics are stored)
    sync_metadata_with_qdrant(embedding_service, pain_filtered_documents)

    # Configure noise detection parameters
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "reddit_pain_documents"
    EPS = 0.5
    MIN_SAMPLES = 5

    FEATURE_PAYLOAD_KEYS = [
        ["pain_metrics", "pain_score"],
        "score",
        "upvote_ratio",
        "num_comments",
        ["pain_metrics", "post_pain", "sentiment", "compound"],
        ["pain_metrics", "post_pain", "keyword_matches", "frustration"],
        ["pain_metrics", "post_pain", "total_keyword_matches"],
        ["quality_metrics", "overall_quality"],
        ["quality_metrics", "engagement_score"],
        ["quality_metrics", "discussion_depth"]
    ]

    # Perform noise detection
    print("Performing noise detection...")
    noise_detector = NoiseDetector(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        feature_payload_keys=FEATURE_PAYLOAD_KEYS
    )

    all_doc_ids, all_features, all_doc_payloads, cluster_labels = noise_detector.detect_noise()

    if not all_doc_ids:
        print("Exiting due to no data after noise detection retrieval.")
        exit()

    # Identify noise points
    noise_point_indices = np.where(cluster_labels == -1)[0]
    noise_document_ids = [all_doc_ids[i] for i in noise_point_indices]

    # Update Qdrant with noise flags
    noise_detector.update_noise_flags_in_qdrant(noise_document_ids, all_doc_ids)

    # Print summary
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))

    print(f"\nPipeline Summary:")
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"After quality filtering: {len(quality_filtered_documents)}")
    print(f"After pain filtering: {len(pain_filtered_documents)}")
    print(f"Noise points detected: {len(noise_document_ids)}")
    print(f"Cluster distribution: {cluster_distribution}")
    print("Noise detection and Qdrant payload update complete.")

    print("\nPipeline execution complete. Starting interactive search...")
    embedding_service = QdrantEmbeddingService()
    interactive_search(embedding_service)