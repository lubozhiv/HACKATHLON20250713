import json
import os
from typing import List, Dict, Any
from langchain.schema import Document
from AdvancedPreprocessingImplementation.no_2_filter_2_pain_detection import AdvancedPainDetector
from archive.archived_filters.no_4_filter_3_noise import NoiseDetector
import numpy as np
from embedded_to_qdrant import QdrantEmbeddingService   # MY EMBEDDEDINGS



def json_to_langchain_documents(json_data: List[Dict[str, Any]], filename: str = "") -> List[Document]:
    """
    Convert JSON topics with comments to LangChain Document objects.

    Args:
        json_data: List of topic dictionaries with the specified structure
        filename: Source filename to include in metadata

    Returns:
        List of LangChain Document objects
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
            if body:  # Only add non-empty comment bodies
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
            "source_file": filename  # Add the source filename to metadata
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

    Args:
        file_path: Path to the JSON file

    Returns:
        List of topic dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Handle both single topic and list of topics
    if isinstance(data, dict):
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON data must be a dictionary or list of dictionaries")


def load_all_json_files_from_directory(directory: str) -> List[Document]:
    """
    Load all JSON files from a directory and convert them to LangChain Documents.

    Args:
        directory: Path to the directory containing JSON files

    Returns:
        Combined list of all LangChain Document objects from all files
    """
    all_documents = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            try:
                # Load JSON data from file
                json_data = load_json_from_file(file_path)
                # Convert to documents and add filename metadata
                documents = json_to_langchain_documents(json_data, filename)
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    return all_documents



if __name__ == "__main__":
    # Load all documents from the datasets directory
    all_documents = load_all_json_files_from_directory('datasets')


    pain_detector = AdvancedPainDetector()
    pain_filtered_docs_filter_2 = pain_detector.filter_documents_by_pain(
        all_documents,
        min_pain_score=0.01, #why this score should be set that low
        max_pain_score=1.0
    )
    # Print summary information
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"Documents after quality filtering: {len(pain_filtered_docs_filter_2)}")
    print(f"Filtered {len(all_documents) - len(pain_filtered_docs_filter_2)} low-quality documents")

    # EMBEDDINGS TO Qdrant
    embedding_service = QdrantEmbeddingService()
    embedding_service.create_collection(recreate=False)
    embedding_service.insert_documents(pain_filtered_docs_filter_2)

    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "reddit_pain_documents"  # This should match the collection name used by QdrantEmbeddingService

    EPS = 0.5
    MIN_SAMPLES = 5

    FEATURE_PAYLOAD_KEYS = [
        ["pain_metrics", "pain_score"],
        "score",  # Top-level score
        "upvote_ratio",  # Top-level upvote_ratio
        "num_comments",  # Top-level num_comments
        ["pain_metrics", "post_pain", "sentiment", "compound"],
        ["pain_metrics", "post_pain", "keyword_matches", "frustration"],
        ["pain_metrics", "post_pain", "total_keyword_matches"],
        # Add other numerical fields you want to use for noise detection
    ]


    noise_detector = NoiseDetector(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        feature_payload_keys=FEATURE_PAYLOAD_KEYS
    )

    # Perform noise detection using the detector instance
    all_doc_ids, all_features, all_doc_payloads, cluster_labels = noise_detector.detect_noise()

    if not all_doc_ids:
        print("Exiting due to no data after noise detection retrieval.")
        exit()

    # Identify noise points from the cluster labels
    noise_point_indices = np.where(cluster_labels == -1)[0]
    noise_document_ids = [all_doc_ids[i] for i in noise_point_indices]

    # Update Qdrant with the noise flags
    noise_detector.update_noise_flags_in_qdrant(noise_document_ids, all_doc_ids)

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))
    print("Noise detection and Qdrant payload update complete.")


    # Print first few filtered documents if you want to inspect them
    # for i, doc in enumerate(pain_filtered_docs_filter_2[:200], 1):
    #     print(f"\nDocument {i}:")
    #     print(f"Source file: {doc.metadata['source_file']}")
    #     # print(f"Quality Score: {doc.metadata['quality_metrics']['overall_quality']:.2f}")
    #     print(f"Page Content (first 200 chars):\n{doc.page_content[:200]}...")
    #     print(f"Metadata: {doc.metadata}")
    #     print("-" * 50)