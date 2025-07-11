
**- Run Qdrant Server Locally with Docker**

docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
########################################################
pip install vaderSentiment
