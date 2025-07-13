import json
import os
from typing import List, Dict, Any
from langchain.schema import Document


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


# Example usage
if __name__ == "__main__":
    # Load all documents from the datasets directory
    all_documents = load_all_json_files_from_directory('datasets')

    # Apply quality filter
    quality_threshold = 0.5  # Adjust this value as needed
    filtered_documents = filter_by_quality(all_documents, min_score=quality_threshold)

    # Print summary information
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"Documents after quality filtering: {len(filtered_documents)}")
    print(f"Filtered {len(all_documents) - len(filtered_documents)} low-quality documents")

    # Print first few filtered documents if you want to inspect them
    for i, doc in enumerate(filtered_documents[:200], 1):
        print(f"\nDocument {i}:")
        print(f"Source file: {doc.metadata['source_file']}")
        print(f"Quality Score: {doc.metadata['quality_metrics']['overall_quality']:.2f}")
        print(f"Page Content (first 200 chars):\n{doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)