import json
import os
from typing import List, Dict, Any
from langchain.schema import Document
from AdvancedPreprocessingImplementation.no_1_filter_1_quality_metrics import filter_by_quality
from AdvancedPreprocessingImplementation.no_2_filter_2_pain_detection import AdvancedPainDetector
from AdvancedPreprocessingImplementation.no_3_filter_3_noise_filteringV2 import RedditNoiseFilter



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


def apply_all_filters(documents: List[Document],
                      quality_threshold: float = 0.5,
                      min_pain_score: float = 0.01,
                      max_pain_score: float = 1.0,
                      noise_threshold: float = 0.3) -> List[Document]:
    """
    Apply all three filters to the documents in sequence:
    1. Quality filter
    2. Pain detection filter
    3. Noise filter

    Args:
        documents: List of input documents
        quality_threshold: Minimum quality score to keep
        min_pain_score: Minimum pain score to keep
        max_pain_score: Maximum pain score to keep
        noise_threshold: Similarity threshold for comment filtering

    Returns:
        List of filtered documents
    """
    # Initialize filters
    pain_detector = AdvancedPainDetector()
    noise_filter = RedditNoiseFilter(noise_threshold=noise_threshold)

    # Apply filters in sequence
    quality_filtered = filter_by_quality(documents, min_score=quality_threshold)
    pain_filtered = pain_detector.filter_documents_by_pain(
        quality_filtered,
        min_pain_score=min_pain_score,
        max_pain_score=max_pain_score
    )
    noise_filtered = noise_filter.filter_documents_by_noise(pain_filtered)

    return noise_filtered



# Example usage
if __name__ == "__main__":
    # Load all documents from the datasets directory
    all_documents = load_all_json_files_from_directory('datasets')

    # Apply all filters
    filtered_documents = apply_all_filters(
        all_documents,
        quality_threshold=0.3,
        min_pain_score=0.002,
        max_pain_score=1.0,
        noise_threshold=0.25
    )

    # Print summary information
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"Documents after all filtering: {len(filtered_documents)}")
    print(f"Filtered {len(all_documents) - len(filtered_documents)} documents")

    # Print first few filtered documents if you want to inspect them
    for i, doc in enumerate(filtered_documents[:5], 1):
        print(f"\nDocument {i}:")
        print(f"Source file: {doc.metadata['source_file']}")
        print(f"Page Content (first 200 chars):\n{doc.page_content[:200]}...")
        print("-" * 50)