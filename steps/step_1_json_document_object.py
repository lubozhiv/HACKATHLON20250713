import json
from typing import List, Dict, Any
from langchain.schema import Document


def json_to_langchain_documents(json_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert JSON topics with comments to LangChain Document objects.

    Args:
        json_data: List of topic dictionaries with the specified structure

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
            "comments_extracted": topic.get("comments_extracted", 0)
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


# Example usage
if __name__ == "__main__":
    # Example 1: Using sample data
    sample_data = [
        {
            "id": "example_1",
            "title": "Example Topic",
            "selftext": "This is the main content of the topic.",
            "url": "https://example.com",
            "score": 150,
            "upvote_ratio": 0.95,
            "num_comments": 3,
            "created_utc": 1640995200,
            "author": "user123",
            "subreddit": "example",
            "permalink": "/r/example/comments/example_1/",
            "is_self": True,
            "distinguished": None,
            "stickied": False,
            "over_18": False,
            "spoiler": False,
            "locked": False,
            "comments": [
                {
                    "id": "comment_1",
                    "body": "This is the first comment.",
                    "score": 10,
                    "created_utc": 1640995300,
                    "author": "commenter1",
                    "is_submitter": False,
                    "parent_id": "example_1",
                    "depth": 0
                },
                {
                    "id": "comment_2",
                    "body": "This is the second comment with more details.",
                    "score": 5,
                    "created_utc": 1640995400,
                    "author": "commenter2",
                    "is_submitter": False,
                    "parent_id": "example_1",
                    "depth": 0
                }
            ],
            "comments_extracted": 2
        },
        {
            "id": "example_2",
            "title": "Example Topic",
            "selftext": "This is the main content of the topic.",
            "url": "https://example.com",
            "score": 150,
            "upvote_ratio": 0.95,
            "num_comments": 3,
            "created_utc": 1640995200,
            "author": "user123",
            "subreddit": "example",
            "permalink": "/r/example/comments/example_1/",
            "is_self": True,
            "distinguished": None,
            "stickied": False,
            "over_18": False,
            "spoiler": False,
            "locked": False,
            "comments": [
                {
                    "id": "comment_1",
                    "body": "This is the first comment.",
                    "score": 10,
                    "created_utc": 1640995300,
                    "author": "commenter1",
                    "is_submitter": False,
                    "parent_id": "example_1",
                    "depth": 0
                },
                {
                    "id": "comment_2",
                    "body": "This is the second comment with more details.",
                    "score": 5,
                    "created_utc": 1640995400,
                    "author": "commenter2",
                    "is_submitter": False,
                    "parent_id": "example_1",
                    "depth": 0
                }
            ],
            "comments_extracted": 2
        }

    ]

    # Convert to LangChain documents
    documents = load_json_from_file('datasets/reddit_ChatGPTCoding_hot_500.json')

    # Print results
    for i, doc in enumerate(documents, 1):
        print(f"Document {i}:")
        print(f"Page Content:\n{doc.page_content}")
        print(f"\nMetadata: {doc.metadata}")
        print("-" * 50)

    # Example 2: Loading from file
    # documents = json_to_langchain_documents(load_json_from_file("topics.json"))

    # Example 3: Filtering documents by subreddit (useful for Qdrant)
    # filtered_docs = [doc for doc in documents if doc.metadata["subreddit"] == "example"]

    # Example 4: Filtering by score threshold
    # high_score_docs = [doc for doc in documents if doc.metadata["score"] > 100]