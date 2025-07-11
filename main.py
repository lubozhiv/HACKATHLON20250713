import json
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings


def embed_reddit_posts_to_qdrant(
        json_path: str,
        collection_name: str = "reddit_posts",
        embedding_model: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
) -> bool:
    """
    Load Reddit posts from JSON file and embed them into Qdrant vector store.

    Args:
        json_path: Path to JSON file containing Reddit posts data
        collection_name: Name of the Qdrant collection
        embedding_model: Embedding model to use (default: text-embedding-ada-002)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load JSON data
        with open(json_path, "r") as f:
            posts = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return False

    # Initialize Qdrant client
    qdrant_client = QdrantClient("http://localhost:6333")

    # Check if collection exists, create if not
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE,
            ),
        )

    # Process each post
    points = []
    for post in posts:
        # Prepare text to embed (title + selftext + all comment bodies)
        text_to_embed = f"Title: {post.get('title', '')}\n"
        text_to_embed += f"Post: {post.get('selftext', '')}\n"

        # Add all comments
        for comment in post.get('comments', []):
            text_to_embed += f"Comment: {comment.get('body', '')}\n"

        # Generate embedding
        embedding = embedding_model.embed_query(text_to_embed)

        # Prepare metadata (all fields except comments)
        metadata = {
            k: v for k, v in post.items()
            if k not in ['comments', 'selftext']
        }

        # Add some comment stats to metadata
        metadata['comments_count'] = len(post.get('comments', []))
        metadata['top_comment_score'] = max(
            [c.get('score', 0) for c in post.get('comments', [])],
            default=0
        )

        # Create point for Qdrant
        point = PointStruct(
            id=post['id'],  # Using Reddit post ID as Qdrant ID
            vector=embedding,
            payload={
                "text": text_to_embed,
                "metadata": metadata,
                "original_data": post  # Store complete original data if needed
            }
        )
        points.append(point)

    # Upload to Qdrant in batches
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
        print(f"Successfully embedded {len(points)} posts to Qdrant")
        return True
    except Exception as e:
        print(f"Error uploading to Qdrant: {e}")
        return False