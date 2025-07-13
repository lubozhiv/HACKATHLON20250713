# qdrant_search_idea_generator.py
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from AdvancedPreprocessingImplementation.no_4_embedded_to_qdrant import QdrantEmbeddingService
import os
import dotenv

dotenv.load_dotenv()


class QdrantSearchAndIdeaGenerator:
    def __init__(self, embedding_service: QdrantEmbeddingService):
        self.embedding_service = embedding_service
        self.llm =ChatOpenAI(
            temperature=0.7,
            model=os.getenv("OPENAI_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def search_qdrant(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search Qdrant with the user query and return ranked results.
        """
        # Get the embedding for the query
        query_embedding = self.embedding_service.model.embed_query(query)

        # Search Qdrant with the query embedding
        search_results = self.embedding_service.qdrant_client.search(
            collection_name=self.embedding_service.collection_name,
            query_vector=query_embedding,
            limit=limit * 3,  # Get more initially to filter
            with_payload=True,
            with_vectors=False
        )

        # Process and rank results
        ranked_results = []
        for result in search_results:
            payload = result.payload
            score = float(result.score)  # Ensure score is float

            # Safely extract metrics with defaults
            quality_metrics = payload.get("quality_metrics", {})
            pain_metrics = payload.get("pain_metrics", {})

            quality_score = float(quality_metrics.get("overall_quality", 0))
            pain_score = float(pain_metrics.get("pain_score", 0))
            is_noise = bool(payload.get("noise_detection", False))

            # Apply weights (ensure all values are numeric)
            weighted_score = (
                    0.5 * score +  # Vector similarity
                    0.35 * quality_score +  # Quality
                    0.10 * pain_score -  # Pain
                    0.05 * (1.0 if is_noise else 0.0)  # Noise penalty
            )

            ranked_results.append({
                "id": result.id,
                "score": weighted_score,
                "payload": payload,
                "original_score": score,
                "quality_score": quality_score,
                "pain_score": pain_score,
                "is_noise": is_noise
            })

        # Sort by weighted score and take top results (outside the loop)
        ranked_results.sort(key=lambda x: x["score"], reverse=True)
        return ranked_results[:limit]

    def generate_product_ideas(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate product ideas based on search results.
        """
        # Prepare context from search results
        context = []
        for i, result in enumerate(search_results, 1):
            context.append(
                f"Result {i} (Score: {result['score']:.2f}, Quality: {result['quality_score']:.2f}, "
                f"Pain: {result['pain_score']:.2f}):\n{result['payload']['text']}\n"
                f"---\n"
            )

        context_str = "\n".join(context)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a product ideation assistant. Based on user queries and relevant Reddit discussions about pain points, suggest potential product features or solutions.

The user query is: {query}

Here are relevant discussions ranked by importance (considering content similarity, quality, and pain points):
{context}

Instructions:
1. Analyze the pain points and needs expressed in these discussions
2. Propose 3-5 specific product ideas or features that could address these needs
3. For each idea, explain which pain points it addresses and how
4. Rank the ideas by potential impact and feasibility
5. Keep each idea concise but specific"""),
            ("human", "Please generate product ideas based on the above context."),
            ("human", "Propose a hypothetical user target group.")
        ])

        # Format the prompt
        formatted_prompt = prompt.format_messages(query=query, context=context_str)

        # Generate response
        response = self.llm.invoke(formatted_prompt)
        return response.content

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return both search results and product ideas.
        """
        search_results = self.search_qdrant(query)
        if not search_results:
            return {
                "search_results": [],
                "ideas": "No relevant results found to generate ideas from."
            }

        ideas = self.generate_product_ideas(query, search_results)

        return {
            "search_results": search_results,
            "ideas": ideas
        }