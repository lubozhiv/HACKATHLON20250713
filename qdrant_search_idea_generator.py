# qdrant_search_idea_generator.py
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from AdvancedPreprocessingImplementation.no_4_embedded_to_qdrant import QdrantEmbeddingService
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
import os
import dotenv
import logging

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


class QdrantSearchAndIdeaGenerator:
    def __init__(self, embedding_service: QdrantEmbeddingService):
        self.embedding_service = embedding_service
        self.llm = ChatOpenAI(
            temperature=0.7,
            model=os.getenv("OPENAI_MODEL"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def create_quality_filter(self,
                              min_comment_quality: float = 0.3,
                              min_filtered_comments: int = 1,
                              max_removal_rate: float = 0.9,
                              require_technical: bool = False) -> Filter:
        """
        Create quality-based filters for search results.

        Args:
            min_comment_quality: Minimum comment similarity score
            min_filtered_comments: Minimum number of comments after filtering
            max_removal_rate: Maximum allowed comment removal rate
            require_technical: Whether to require technical content
        """
        conditions = []

        # Quality threshold
        if min_comment_quality > 0:
            conditions.append(
                FieldCondition(key="comment_quality_score", range=Range(gte=min_comment_quality))
            )

        # Minimum filtered comments
        if min_filtered_comments > 0:
            conditions.append(
                FieldCondition(key="filtered_comment_count", range=Range(gte=min_filtered_comments))
            )

        # Maximum removal rate (to avoid over-filtered posts)
        if max_removal_rate < 1.0:
            conditions.append(
                FieldCondition(key="comment_removal_rate", range=Range(lte=max_removal_rate))
            )

        # Technical content requirement
        if require_technical:
            conditions.append(
                FieldCondition(key="has_technical_content", match=MatchValue(value=True))
            )

        return Filter(must=conditions) if conditions else None

    def search_qdrant(self,
                      query: str,
                      limit: int = 20,
                      quality_filter: bool = True,
                      min_comment_quality: float = 0.3,
                      require_technical: bool = False) -> List[Dict[str, Any]]:
        """
        Search Qdrant with the user query and return ranked results.
        Enhanced with quality filtering based on the improved noise filtering.
        """
        try:
            # Get the embedding for the query
            query_embedding = self.embedding_service.model.embed_query(query)

            # Create quality filter if requested
            search_filter = None
            if quality_filter:
                search_filter = self.create_quality_filter(
                    min_comment_quality=min_comment_quality,
                    min_filtered_comments=1,
                    max_removal_rate=0.95,
                    require_technical=require_technical
                )

            # Search Qdrant with the query embedding
            search_params = {
                "collection_name": self.embedding_service.collection_name,
                "query_vector": query_embedding,
                "limit": limit * 3,  # Get more initially to re-rank
                "with_payload": True,
                "with_vectors": False
            }

            if search_filter:
                search_params["query_filter"] = search_filter

            search_results = self.embedding_service.qdrant_client.search(**search_params)

            # Process and rank results with enhanced scoring
            ranked_results = []
            for result in search_results:
                payload = result.payload
                score = float(result.score)

                # Extract enhanced quality metrics
                comment_quality = float(payload.get("comment_quality_score", 0.0))
                filtered_comment_count = int(payload.get("filtered_comment_count", 0))
                original_comment_count = int(payload.get("original_comment_count", 0))
                removal_rate = float(payload.get("comment_removal_rate", 0.0))
                has_technical = bool(payload.get("has_technical_content", False))
                adaptive_threshold = float(payload.get("adaptive_threshold", 0.0))
                word_count = int(payload.get("word_count", 0))

                # Calculate engagement score (more comments = more engagement)
                engagement_score = min(1.0, filtered_comment_count / 10.0)  # Normalize to 0-1

                # Calculate content richness (balanced removal rate indicates good filtering)
                content_richness = 1.0 - abs(0.3 - removal_rate) if removal_rate <= 0.8 else 0.5

                # Technical content bonus
                technical_bonus = 1.2 if has_technical else 1.0

                # Adaptive threshold quality (higher threshold suggests more selective filtering)
                threshold_quality = min(1.0, adaptive_threshold * 2)  # Normalize

                # Content length factor
                length_factor = min(1.0, word_count / 200.0)  # Normalize to reasonable length

                # Calculate comprehensive weighted score
                weighted_score = (
                                         0.35 * score +  # Vector similarity (primary)
                                         0.25 * comment_quality +  # Comment relevance quality
                                         0.15 * engagement_score +  # Community engagement
                                         0.10 * content_richness +  # Filtering quality
                                         0.08 * threshold_quality +  # Adaptive filtering quality
                                         0.07 * length_factor  # Content substance
                                 ) * technical_bonus

                ranked_results.append({
                    "id": result.id,
                    "score": weighted_score,
                    "payload": payload,
                    "metrics": {
                        "original_score": score,
                        "comment_quality": comment_quality,
                        "engagement_score": engagement_score,
                        "content_richness": content_richness,
                        "has_technical": has_technical,
                        "filtered_comments": filtered_comment_count,
                        "original_comments": original_comment_count,
                        "removal_rate": removal_rate,
                        "adaptive_threshold": adaptive_threshold,
                        "word_count": word_count
                    }
                })

            # Sort by weighted score and take top results
            ranked_results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Search for '{query}' returned {len(ranked_results)} results after filtering and ranking")

            return ranked_results[:limit]

        except Exception as e:
            logger.error(f"Error in search_qdrant: {str(e)}")
            return []

    def analyze_search_quality(self, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the quality of search results based on filtering metrics.
        """
        if not search_results:
            return {"quality": "No results", "metrics": {}}

        metrics = [result["metrics"] for result in search_results]

        quality_analysis = {
            "result_count": len(search_results),
            "avg_comment_quality": sum(m["comment_quality"] for m in metrics) / len(metrics),
            "avg_engagement": sum(m["engagement_score"] for m in metrics) / len(metrics),
            "avg_content_richness": sum(m["content_richness"] for m in metrics) / len(metrics),
            "technical_content_ratio": sum(1 for m in metrics if m["has_technical"]) / len(metrics),
            "avg_removal_rate": sum(m["removal_rate"] for m in metrics) / len(metrics),
            "avg_filtered_comments": sum(m["filtered_comments"] for m in metrics) / len(metrics),
            "avg_word_count": sum(m["word_count"] for m in metrics) / len(metrics)
        }

        # Determine overall quality
        overall_quality = (
                quality_analysis["avg_comment_quality"] * 0.4 +
                quality_analysis["avg_engagement"] * 0.3 +
                quality_analysis["avg_content_richness"] * 0.3
        )

        if overall_quality >= 0.7:
            quality_level = "High"
        elif overall_quality >= 0.5:
            quality_level = "Medium"
        else:
            quality_level = "Low"

        return {
            "quality": quality_level,
            "overall_score": overall_quality,
            "metrics": quality_analysis
        }

    def generate_product_ideas(self,
                               query: str,
                               search_results: List[Dict[str, Any]],
                               focus_area: str = "general") -> str:
        """
        Generate product ideas based on search results with enhanced context analysis.
        """
        if not search_results:
            return "No relevant results found to generate ideas from."

        # Analyze search quality
        quality_analysis = self.analyze_search_quality(search_results)

        # Prepare enhanced context from search results
        context_parts = []
        for i, result in enumerate(search_results[:10], 1):  # Limit to top 10 for context
            metrics = result["metrics"]
            payload = result["payload"]

            # Extract key information
            title = payload.get("title", "No title")
            text_preview = payload.get("text", "")[:500] + "..." if len(payload.get("text", "")) > 500 else payload.get(
                "text", "")

            context_parts.append(
                f"Result {i} (Score: {result['score']:.2f}):\n"
                f"Title: {title}\n"
                f"Quality Metrics: Comment Quality={metrics['comment_quality']:.2f}, "
                f"Engagement={metrics['engagement_score']:.2f}, "
                f"Technical Content={metrics['has_technical']}, "
                f"Filtered Comments={metrics['filtered_comments']}/{metrics['original_comments']}\n"
                f"Content: {text_preview}\n"
                f"---\n"
            )

        context_str = "\n".join(context_parts)

        # Create enhanced prompt template based on focus area
        if focus_area == "technical":
            system_prompt = """You are a technical product ideation assistant specializing in developer tools and technical solutions. Based on user queries and high-quality Reddit discussions about technical pain points, suggest specific product features or technical solutions.

The user query is: {query}

Search Quality Analysis: {quality_analysis}

Here are relevant technical discussions ranked by importance (considering content similarity, comment quality, and technical relevance):
{context}

Instructions:
1. Focus on technical pain points and developer/technical user needs
2. Propose 3-5 specific technical product ideas or features
3. For each idea, explain the technical problem it solves and implementation approach
4. Consider API design, developer experience, and technical feasibility
5. Rank ideas by technical impact and development complexity
6. Include considerations for technical adoption and integration"""

        elif focus_area == "business":
            system_prompt = """You are a business-focused product ideation assistant. Based on user queries and high-quality Reddit discussions about business and market pain points, suggest product opportunities and business solutions.

The user query is: {query}

Search Quality Analysis: {quality_analysis}

Here are relevant business discussions ranked by importance (considering content similarity, comment quality, and market relevance):
{context}

Instructions:
1. Analyze business pain points and market opportunities
2. Propose 3-5 specific business-focused product ideas
3. For each idea, explain the market need, target customer, and business model potential
4. Consider market size, competition, and go-to-market strategy
5. Rank ideas by market opportunity and business viability
6. Include monetization and scaling considerations"""

        else:  # general
            system_prompt = """You are a comprehensive product ideation assistant. Based on user queries and high-quality Reddit discussions about user pain points, suggest potential product features or solutions.

The user query is: {query}

Search Quality Analysis: {quality_analysis}

Here are relevant discussions ranked by importance (considering content similarity, comment quality, engagement, and content richness):
{context}

Instructions:
1. Analyze the pain points and needs expressed in these high-quality discussions
2. Propose 3-5 specific product ideas or features that could address these needs
3. For each idea, explain which pain points it addresses and how
4. Consider user experience, technical feasibility, and market potential
5. Rank the ideas by potential impact and implementation feasibility
6. Keep each idea concise but specific with actionable details"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please generate product ideas based on the above context."),
            ("human", "Also propose a target user group and key success metrics.")
        ])

        # Format the prompt
        formatted_prompt = prompt.format_messages(
            query=query,
            context=context_str,
            quality_analysis=f"Quality Level: {quality_analysis['quality']}, "
                             f"Overall Score: {quality_analysis['overall_score']:.2f}, "
                             f"Avg Comment Quality: {quality_analysis['metrics']['avg_comment_quality']:.2f}, "
                             f"Technical Content: {quality_analysis['metrics']['technical_content_ratio']:.1%}"
        )

        # Generate response
        try:
            response = self.llm.invoke(formatted_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Error generating product ideas: {str(e)}")
            return f"Error generating ideas: {str(e)}"

    def process_query(self,
                      query: str,
                      focus_area: str = "general",
                      quality_filter: bool = True,
                      min_comment_quality: float = 0.3,
                      require_technical: bool = False) -> Dict[str, Any]:
        """
        Process a user query and return both search results and product ideas.
        Enhanced with quality filtering and focus area specification.
        """
        # Adjust parameters based on focus area
        if focus_area == "technical":
            require_technical = True
            min_comment_quality = max(min_comment_quality, 0.4)  # Higher quality for technical content

        search_results = self.search_qdrant(
            query=query,
            limit=20,
            quality_filter=quality_filter,
            min_comment_quality=min_comment_quality,
            require_technical=require_technical
        )

        if not search_results:
            return {
                "search_results": [],
                "ideas": "No relevant results found to generate ideas from. Try adjusting quality filters or broadening the search terms.",
                "quality_analysis": {"quality": "No results", "metrics": {}},
                "recommendations": [
                    "Try lowering the minimum comment quality threshold",
                    "Remove technical content requirement if enabled",
                    "Use broader or different search terms",
                    "Check if the collection contains relevant data"
                ]
            }

        quality_analysis = self.analyze_search_quality(search_results)
        ideas = self.generate_product_ideas(query, search_results, focus_area)

        # Generate recommendations based on search quality
        recommendations = []
        if quality_analysis["overall_score"] < 0.5:
            recommendations.append("Consider refining search terms for better quality results")
        if quality_analysis["metrics"]["technical_content_ratio"] < 0.3 and focus_area == "technical":
            recommendations.append("Low technical content ratio - consider broader technical terms")
        if quality_analysis["metrics"]["avg_engagement"] < 0.3:
            recommendations.append("Low engagement results - try more popular or trending topics")

        return {
            "search_results": search_results,
            "ideas": ideas,
            "quality_analysis": quality_analysis,
            "focus_area": focus_area,
            "recommendations": recommendations if recommendations else ["Search results look good!"]
        }

    def get_trending_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get trending topics based on high-quality, high-engagement posts.
        """
        try:
            # Create filter for high-quality, engaging content
            trending_filter = Filter(must=[
                FieldCondition(key="comment_quality_score", range=Range(gte=0.5)),
                FieldCondition(key="filtered_comment_count", range=Range(gte=5)),
                FieldCondition(key="comment_removal_rate", range=Range(lte=0.7))
            ])

            # Get recent high-quality posts
            results = self.embedding_service.qdrant_client.scroll(
                collection_name=self.embedding_service.collection_name,
                scroll_filter=trending_filter,
                limit=limit * 2,  # Get more to select best
                with_payload=True,
                with_vectors=False
            )

            trending_topics = []
            for point in results[0][:limit]:
                payload = point.payload
                trending_topics.append({
                    "title": payload.get("title", "No title"),
                    "engagement_score": payload.get("filtered_comment_count", 0),
                    "quality_score": payload.get("comment_quality_score", 0.0),
                    "has_technical": payload.get("has_technical_content", False),
                    "preview": payload.get("text", "")[:200] + "..." if len(
                        payload.get("text", "")) > 200 else payload.get("text", "")
                })

            return trending_topics

        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # This would be used with an actual QdrantEmbeddingService instance
    pass