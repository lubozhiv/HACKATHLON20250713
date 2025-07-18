from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple, Set
from langchain.schema import Document
import numpy.typing as npt
import re


class RedditNoiseFilter:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', noise_threshold: float = 0.51):
        """
        Initialize the noise filter with a sentence transformer model and threshold.

        Args:
            model_name: Name of the sentence transformer model to use
            noise_threshold: Similarity threshold below which comments are considered noise (0-1)
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.noise_threshold = noise_threshold

        # Common spam patterns for Reddit
        self.spam_patterns = [
            'check out my', 'click here', 'buy now', 'limited offer',
            'make money', 'work from home', 'telegram', 'whatsapp',
            'dm me', 'porn', 'onlyfans', '18+', 'nsfw'
        ]

        # Technical terms common in coding subreddits
        self.technical_terms = {
            'api', 'code', 'function', 'class', 'method', 'variable',
            'bug', 'error', 'debug', 'compile', 'runtime', 'syntax',
            'algorithm', 'data', 'structure', 'array', 'list', 'dict',
            'github', 'git', 'commit', 'pull', 'push', 'merge',
            'python', 'javascript', 'java', 'cpp', 'rust', 'golang',
            'react', 'vue', 'angular', 'node', 'npm', 'pip',
            'docker', 'kubernetes', 'aws', 'cloud', 'server',
            'database', 'sql', 'nosql', 'mongodb', 'postgres',
            'ai', 'ml', 'llm', 'gpt', 'claude', 'gemini', 'model',
            'prompt', 'embedding', 'vector', 'transformer',
            'mcp', 'cli', 'gui', 'ui', 'ux', 'frontend', 'backend'
        }

    def embed_text(self, text: str) -> npt.NDArray:
        """Generate embedding for a single text"""
        return self.model.encode(text, convert_to_numpy=True)

    def calculate_similarity(self, embedding1: npt.NDArray, embedding2: npt.NDArray) -> float:
        """Calculate cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)))

    def extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'been',
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                      'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                      'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'them',
                      'my', 'your', 'his', 'her', 'its', 'our', 'their'}

        # Keep words that are technical terms or longer than 3 characters
        keywords = {word for word in words
                    if (word not in stop_words and len(word) > 3) or word in self.technical_terms}

        return keywords

    def contains_technical_vocabulary(self, text: str) -> bool:
        """Check if text contains technical coding-related terms"""
        text_lower = text.lower()
        return any(term in text_lower for term in self.technical_terms)

    def is_spam_or_offtopic(self, comment_text: str, post_keywords: Set[str]) -> bool:
        """Check if comment is likely spam or completely off-topic"""
        comment_lower = comment_text.lower()

        # Check for spam patterns
        if any(pattern in comment_lower for pattern in self.spam_patterns):
            return True

        # Check for keyword overlap
        comment_keywords = self.extract_keywords(comment_text)
        if post_keywords and comment_keywords:
            keyword_overlap = len(comment_keywords & post_keywords) / len(post_keywords)
            if keyword_overlap < 0.05:  # Less than 5% overlap
                return True

        return False

    def calculate_adaptive_threshold(self, similarities: List[float]) -> float:
        """Calculate adaptive threshold based on similarity distribution"""
        if not similarities:
            return self.noise_threshold

        # Use statistical measures to set threshold
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)

        # Keep comments within 1.5 standard deviations below mean
        # This adapts to the conversation's natural variance
        adaptive_threshold = max(0.3, mean_sim - 1.5 * std_sim)

        # Don't exceed the configured threshold
        return min(adaptive_threshold, self.noise_threshold)

    def calculate_relevance_score(self,
                                  comment: Dict,
                                  post_embedding: npt.NDArray,
                                  post_text: str,
                                  prev_comment_embedding: npt.NDArray = None,
                                  comment_position: int = 0,
                                  total_comments: int = 1) -> float:
        """
        Calculate comprehensive relevance score using multiple signals
        """
        comment_text = comment.get('body', '')
        if not comment_text.strip():
            return 0.0

        # Generate embedding for comment
        comment_embedding = self.embed_text(comment_text)

        # 1. Semantic similarity to post
        post_similarity = self.calculate_similarity(post_embedding, comment_embedding)
        normalized_post_sim = max(0, min(1, (post_similarity + 1) / 2))

        # 2. Conversation flow - similarity to previous comment
        conversation_flow_score = 0.5  # Default neutral score
        if prev_comment_embedding is not None and comment_position > 0:
            prev_similarity = self.calculate_similarity(prev_comment_embedding, comment_embedding)
            conversation_flow_score = max(0, min(1, (prev_similarity + 1) / 2))

        # 3. Length ratio (avoid very short or very long comments relative to post)
        post_length = max(len(post_text.split()), 10)  # Minimum 10 words
        comment_length = len(comment_text.split())
        length_ratio = comment_length / post_length

        # Optimal length ratio between 0.1 and 3.0
        if 0.1 <= length_ratio <= 3.0:
            length_score = 1.0
        elif length_ratio < 0.1:
            length_score = length_ratio * 10  # Scale up very short comments
        else:
            length_score = max(0.3, 1.0 - (length_ratio - 3.0) * 0.1)

        # 4. Technical content bonus
        has_technical = self.contains_technical_vocabulary(comment_text)
        technical_bonus = 1.2 if has_technical else 1.0

        # 5. Position decay - earlier comments often more relevant
        position_factor = 1.0 - (comment_position / max(total_comments, 1)) * 0.2

        # 6. Check for meta-discussion indicators (common in coding subreddits)
        meta_indicators = ['tried this', 'works for me', 'same issue', 'fixed it',
                           'thanks', 'helped', 'solved', 'problem', 'solution',
                           'update:', 'edit:', 'note:', 'btw', 'fyi']
        has_meta = any(indicator in comment_text.lower() for indicator in meta_indicators)
        meta_bonus = 1.1 if has_meta else 1.0

        # Weighted combination of all factors
        relevance = (
                            0.4 * normalized_post_sim +  # Primary factor
                            0.15 * conversation_flow_score +  # Conversation continuity
                            0.15 * length_score +  # Appropriate length
                            0.1 * position_factor +  # Position in thread
                            0.2  # Base score
                    ) * technical_bonus * meta_bonus

        # Store the embedding for next iteration
        comment['_embedding'] = comment_embedding

        return min(relevance, 1.0)

    def filter_documents_by_noise(self, documents: List[Document]) -> List[Document]:
        """
        Filter documents by removing unrelated comments based on semantic similarity.

        Args:
            documents: List of LangChain Document objects with Reddit data

        Returns:
            List of filtered Document objects with unrelated comments removed
        """
        filtered_docs = []

        for doc in documents:
            # Initialize variables for this document
            filtered_comments = []
            all_similarities = []
            relevance_scores = []

            # Skip if no comments exist
            if 'comments' not in doc.metadata or not doc.metadata['comments']:
                doc.metadata['comment_similarity_score'] = 0.0
                doc.metadata['filtering_stats'] = {
                    'original_count': 0,
                    'filtered_count': 0,
                    'removal_rate': 0.0
                }
                filtered_docs.append(doc)
                continue

            # Combine title and content
            title = doc.metadata.get('title', '')
            content = doc.metadata.get('selftext', '')
            post_text = f"{title} {content}".strip()

            if not post_text:
                doc.metadata['comment_similarity_score'] = 0.0
                doc.metadata['filtering_stats'] = {
                    'original_count': len(doc.metadata['comments']),
                    'filtered_count': 0,
                    'removal_rate': 1.0
                }
                filtered_docs.append(doc)
                continue

            # Extract keywords from post
            post_keywords = self.extract_keywords(post_text)

            # Generate post embedding
            post_embedding = self.embed_text(post_text)
            comments = doc.metadata['comments']
            original_count = len(comments)

            # Track previous comment embedding for conversation flow
            prev_comment_embedding = None

            # First pass: calculate all relevance scores
            for i, comment in enumerate(comments):
                if not comment.get('body', '').strip():
                    continue

                # Check for obvious spam first
                if self.is_spam_or_offtopic(comment['body'], post_keywords):
                    relevance_scores.append(0.0)
                    continue

                # Calculate comprehensive relevance score
                relevance = self.calculate_relevance_score(
                    comment,
                    post_embedding,
                    post_text,
                    prev_comment_embedding,
                    i,
                    original_count
                )

                relevance_scores.append(relevance)
                all_similarities.append(relevance)

                # Update previous embedding if comment has one
                if '_embedding' in comment:
                    prev_comment_embedding = comment['_embedding']

            # Calculate adaptive threshold based on score distribution
            if relevance_scores:
                adaptive_threshold = self.calculate_adaptive_threshold(relevance_scores)
            else:
                adaptive_threshold = self.noise_threshold

            # Second pass: filter based on adaptive threshold
            prev_comment_embedding = None
            for i, comment in enumerate(comments):
                if i < len(relevance_scores) and relevance_scores[i] >= adaptive_threshold:
                    # Remove temporary embedding before adding to filtered list
                    if '_embedding' in comment:
                        prev_comment_embedding = comment.pop('_embedding')
                    filtered_comments.append(comment)

            # Calculate average similarity
            avg_similarity = float(np.mean(all_similarities)) if all_similarities else 0.0

            # Create new document with filtered comments
            new_metadata = doc.metadata.copy()
            new_metadata['comments'] = filtered_comments
            new_metadata['comments_extracted'] = len(filtered_comments)
            new_metadata['comment_similarity_score'] = avg_similarity
            new_metadata['filtering_stats'] = {
                'original_count': original_count,
                'filtered_count': len(filtered_comments),
                'removal_rate': 1.0 - (len(filtered_comments) / original_count if original_count > 0 else 0),
                'adaptive_threshold': adaptive_threshold,
                'score_distribution': {
                    'min': min(relevance_scores) if relevance_scores else 0,
                    'max': max(relevance_scores) if relevance_scores else 0,
                    'mean': np.mean(relevance_scores) if relevance_scores else 0,
                    'std': np.std(relevance_scores) if relevance_scores else 0
                }
            }

            # Rebuild page_content to reflect filtered comments
            page_content_parts = []
            if new_metadata.get('title'):
                page_content_parts.append(f"Title: {new_metadata['title']}")
            if new_metadata.get('selftext'):
                page_content_parts.append(f"Content: {new_metadata['selftext']}")
            if filtered_comments:
                page_content_parts.append("Comments:")
                for i, comment in enumerate(filtered_comments, 1):
                    page_content_parts.append(f"Comment {i}: {comment['body']}")

            filtered_doc = Document(
                page_content="\n\n".join(page_content_parts),
                metadata=new_metadata
            )
            filtered_docs.append(filtered_doc)

        return filtered_docs


# Example usage (for testing)
if __name__ == "__main__":
    # Example document (simplified)
    from langchain.schema import Document

    example_doc = Document(
        page_content="Title: Test\n\nContent: Test content",
        metadata={
            "title": "Space and planets",
            "selftext": "How many planets are there in space",
            "comments": [
                {"body": "Mars has the largest volcano in our solar system called Olympus Mons"},
                {"body": "I love pepperoni pizza on Friday nights"},
                {"body": "Jupiter's Great Red Spot is actually a massive storm that's been raging for centuries"},
                {"body": "My dog needs to go to the vet tomorrow"},
                {"body": "Saturn's rings are made primarily of ice particles and rocky debris"},
                {"body": "The traffic was terrible this morning"},
                {"body": "Venus is the hottest planet in our solar system despite not being closest to the Sun"},
                {"body": "I can't believe how expensive groceries have gotten"},
                {"body": "The asteroid belt lies between Mars and Jupiter"},
                {"body": "This recipe for chocolate chip cookies is amazing"},
                {"body": "Neptune takes 165 Earth years to complete one orbit around the Sun"},
                {"body": "I'm thinking about changing my hairstyle"},
                {"body": "The Moon is slowly moving away from Earth at about 3.8 cm per year"},
                {"body": "My neighbor's cat keeps coming into my yard"},
                {"body": "Pluto was reclassified as a dwarf planet in 2006"},
                {"body": "I need to remember to pay my electricity bill"},
                {"body": "Europa, one of Jupiter's moons, may have an ocean beneath its icy surface"},
                {"body": "The new restaurant downtown has terrible service"},
                {"body": "Mercury has extreme temperature variations between day and night"},
                {"body": "I'm getting tired of all these spam phone calls"},
                {"body": "The Milky Way galaxy contains an estimated 100-400 billion stars"},
                {"body": "My car is making a weird noise when I brake"},
                {"body": "Titan, Saturn's largest moon, has lakes of liquid methane"},
                {"body": "I can't find my keys anywhere"},
                {"body": "The International Space Station orbits Earth approximately every 90 minutes"},
                {"body": "This weather has been so unpredictable lately"},
                {"body": "Uranus rotates on its side, making it unique among planets"},
                {"body": "I'm thinking about adopting a rescue dog"},
                {"body": "The Voyager probes have traveled beyond our solar system"},
                {"body": "My phone battery dies so quickly now"},
                {"body": "Exoplanets are planets that orbit stars outside our solar system"},
                {"body": "I hate doing laundry on weekends"},
                {"body": "The Sun converts 4 million tons of matter into energy every second"},
                {"body": "My favorite TV show got cancelled"},
                {"body": "Mars has seasons similar to Earth due to its axial tilt"},
                {"body": "I need to schedule my annual dentist appointment"},
                {"body": "Black holes have gravitational fields so strong that light cannot escape"},
                {"body": "The coffee shop on Main Street has the best lattes"},
                {"body": "Ganymede is the largest moon in our solar system"},
                {"body": "I'm so tired of political advertisements"},
                {"body": "The Kuiper Belt contains many icy objects beyond Neptune"},
                {"body": "My garden tomatoes aren't growing very well this year"},
                {"body": "Kepler-452b is sometimes called Earth's cousin due to its similarities"},
                {"body": "I forgot to set my alarm clock last night"},
                {"body": "Solar winds can affect satellite communications on Earth"},
                {"body": "The gym was packed during lunch today"},
                {"body": "Ceres is the largest object in the asteroid belt"},
                {"body": "I'm looking forward to the weekend"},
                {"body": "The Parker Solar Probe is studying the Sun's corona up close"},
                {"body": "My washing machine is making strange noises"},
                {"body": "Phobos and Deimos are the two moons of Mars"},
                {"body": "I can't decide what to watch on Netflix tonight"},
                {"body": "The James Webb Space Telescope is revolutionizing our understanding of the universe"},
                {"body": "My coworker brought donuts to the office today"},
                {"body": "Water ice has been discovered on the Moon's poles"},
                {"body": "I'm getting a new laptop next week"},
                {"body": "The Oort Cloud is a theoretical sphere of icy objects at the edge of our solar system"},
                {"body": "My sister is getting married next month"},
                {"body": "Goldilocks zone refers to the habitable region around a star where liquid water can exist"},
                {"body": "I really need to clean my garage"},
                {"body": "Supernovas are explosive deaths of massive stars"},
                {"body": "The line at the grocery store was incredibly long"},
                {"body": "Io is the most volcanically active body in our solar system"},
                {"body": "I'm considering taking up yoga"},
                {"body": "Red giants are stars in a late stage of stellar evolution"},
                {"body": "My favorite restaurant is closing down permanently"},
                {"body": "The atmosphere of Venus is 96% carbon dioxide"},
                {"body": "I lost my umbrella somewhere"},
                {"body": "Proxima Centauri is the closest star to our solar system"},
                {"body": "The construction noise next door starts too early"},
                {"body": "Asteroid impacts may have brought water to early Earth"},
                {"body": "I'm thinking about repainting my living room"},
                {"body": "Neutron stars are incredibly dense remnants of collapsed stars"},
                {"body": "My flight got delayed by three hours"},
                {"body": "The Hubble Space Telescope has been operational for over 30 years"},
                {"body": "I can't believe how much my utility bills have increased"},
                {"body": "Comets are often called dirty snowballs due to their composition"},
                {"body": "My neighbor's dog barks all night long"},
                {"body": "The search for extraterrestrial life often focuses on Mars and Europa"},
                {"body": "I'm trying to eat healthier this year"},
                {"body": "Pulsars are rapidly rotating neutron stars that emit beams of radiation"},
                {"body": "The WiFi at my house keeps cutting out"},
                {"body": "Venus rotates backwards compared to most other planets"},
                {"body": "I'm planning a vacation to Hawaii next summer"},
                {"body": "The rings of Saturn were first observed by Galileo in 1610"},
                {"body": "My subscription to that streaming service is too expensive"},
                {"body": "Light from the Sun takes about 8 minutes to reach Earth"},
                {"body": "I'm learning to play guitar in my spare time"},
                {"body": "The Great Red Spot on Jupiter is larger than Earth"},
                {"body": "My back has been bothering me lately"},
                {"body": "Spacecraft use gravity assists to gain speed and change direction"},
                {"body": "I'm excited about the new season of my favorite show"},
                {"body": "The North Star, Polaris, is not the brightest star in the sky"},
                {"body": "My boss wants me to work overtime this weekend"},
                {"body": "Methane in Titan's atmosphere gives it an orange hue"},
                {"body": "I'm trying to find a good mechanic for my car"},
                {"body": "The Drake Equation estimates the number of communicating civilizations in our galaxy"},
                {"body": "My phone screen cracked when I dropped it"},
                {"body": "Solar eclipses occur when the Moon passes between Earth and the Sun"},
                {"body": "I'm considering switching to a different internet provider"},
                {"body": "The Cassini spacecraft provided incredible data about Saturn before its mission ended"},
                {"body": "My favorite coffee shop changed their hours"},
                {"body": "Enceladus shoots geysers of water ice from its south pole"},
                {"body": "I'm getting new tires for my car next week"},
                {"body": "The Perseverance rover is currently exploring Mars for signs of ancient life"},
                {"body": "I can't find a good parking spot anywhere downtown"}
            ],
            "comments_extracted": 106
        }
    )

    noise_filter = RedditNoiseFilter()
    filtered_docs = noise_filter.filter_documents_by_noise([example_doc])

    print(f"Original comments: {len(example_doc.metadata['comments'])}")
    print(f"Filtered comments: {len(filtered_docs[0].metadata['comments'])}")
    print(f"Filtering stats: {filtered_docs[0].metadata['filtering_stats']}")