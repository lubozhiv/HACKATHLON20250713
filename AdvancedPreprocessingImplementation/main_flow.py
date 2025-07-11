# main_flow.py - Integrated Pain Point Search Engine
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Optional
from dataclasses import dataclass
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models


@dataclass
class QualityMetrics:
    engagement_score: float
    discussion_depth: float
    response_quality: float
    time_pattern_score: float
    overall_quality: float


class CompositeQualityScorer:
    def __init__(self):
        self.weights = {
            'engagement': 0.25,
            'depth': 0.30,
            'response_quality': 0.25,
            'time_pattern': 0.20
        }

    def calculate_engagement_score(self, post_data: Dict) -> float:
        """Updated to work with Reddit API structure"""
        num_comments = post_data.get('num_comments', 0)
        score = post_data.get('score', 0)  # net score (upvotes - downvotes)
        upvote_ratio = post_data.get('upvote_ratio', 1.0)  # percentage of upvotes

        if num_comments == 0:
            return 0.0

        # Estimate controversy (0 = no downvotes, 0.5 = balanced, 1 = heavily downvoted)
        controversy = abs(upvote_ratio - 0.5) * 2 if upvote_ratio < 1.0 else 0

        # New engagement calculation
        base_score = min(num_comments / 20, 1.0)
        participation_ratio = min(num_comments / max(score, 1), 5.0)

        return (base_score + participation_ratio / 5 + controversy) / 3

    def calculate_discussion_depth(self, comments: List[Dict]) -> float:
        """Measure how deep and back-and-forth the discussion gets"""
        if not comments:
            return 0.0

        # Count reply chains (comments with parent_id)
        reply_chains = sum(1 for c in comments if c.get('parent_id'))

        # Average comment length (longer = more thoughtful)
        avg_length = np.mean([len(c.get('body', '')) for c in comments])
        length_score = min(avg_length / 200, 1.0)  # 200+ chars = max score

        # Reply depth score
        depth_score = min(reply_chains / len(comments), 1.0)

        return (length_score + depth_score) / 2

    def calculate_response_quality(self, comments: List[Dict]) -> float:
        """Evaluate the quality of responses using text analysis"""
        if not comments:
            return 0.0

        quality_indicators = [
            'thanks', 'helpful', 'solved', 'exactly', 'perfect',
            'this works', 'great solution', 'fixed it'
        ]

        problem_indicators = [
            'still broken', 'doesn\'t work', 'same issue',
            'not working', 'failed', 'error'
        ]

        positive_responses = 0
        negative_responses = 0

        for comment in comments:
            body = comment.get('body', '').lower()
            if any(indicator in body for indicator in quality_indicators):
                positive_responses += 1
            if any(indicator in body for indicator in problem_indicators):
                negative_responses += 1

        if positive_responses + negative_responses == 0:
            return 0.5  # Neutral if no clear indicators

        return positive_responses / (positive_responses + negative_responses)

    def calculate_time_pattern_score(self, comments: List[Dict]) -> float:
        """Analyze timing patterns for organic vs. spam discussions"""
        if len(comments) < 2:
            return 0.0

        # Get timestamps and sort
        timestamps = []
        for comment in comments:
            if 'created_utc' in comment:
                timestamps.append(comment['created_utc'])

        if len(timestamps) < 2:
            return 0.0

        timestamps.sort()

        # Calculate time gaps between responses
        time_gaps = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # Organic discussions have varied timing (not all at once, not too spread out)
        avg_gap = np.mean(time_gaps)

        # Score based on "natural" conversation timing (1 hour to 1 day is ideal)
        ideal_gap = 3600 * 12  # 12 hours
        if avg_gap < 60:  # Too fast (likely spam/bot)
            return 0.2
        elif avg_gap > 3600 * 48:  # Too slow (likely dead conversation)
            return 0.3
        else:
            # Closer to ideal timing = higher score
            deviation = abs(avg_gap - ideal_gap) / ideal_gap
            return max(0.4, 1.0 - deviation)

    def score_post(self, post_data: Dict, comments: List[Dict]) -> QualityMetrics:
        """Calculate comprehensive quality score for a post"""
        engagement = self.calculate_engagement_score(post_data)
        depth = self.calculate_discussion_depth(comments)
        response_quality = self.calculate_response_quality(comments)
        time_pattern = self.calculate_time_pattern_score(comments)

        # Weighted overall score
        overall = (
                engagement * self.weights['engagement'] +
                depth * self.weights['depth'] +
                response_quality * self.weights['response_quality'] +
                time_pattern * self.weights['time_pattern']
        )

        return QualityMetrics(
            engagement_score=engagement,
            discussion_depth=depth,
            response_quality=response_quality,
            time_pattern_score=time_pattern,
            overall_quality=overall
        )


class AdvancedPainDetector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.pain_keywords = {
            'frustration': ['frustrated', 'annoying', 'hate', 'terrible', 'awful'],
            'problems': ['broken', 'doesn\'t work', 'failing', 'error', 'bug'],
            'wishes': ['wish', 'if only', 'would be nice', 'need', 'want'],
            'impossibility': ['can\'t', 'impossible', 'no way', 'unable to'],
            'comparison': ['better alternative', 'switch to', 'instead of'],
            'time_waste': ['waste of time', 'hours spent', 'tedious', 'manual'],
            'workarounds': ['workaround', 'hack', 'temporary fix', 'band-aid']
        }

        self.pain_patterns = [
            r'why (?:doesn\'t|can\'t|won\'t|isn\'t)',
            r'how (?:do I|can I|to) (?:fix|solve|resolve)',
            r'(?:is there|any) (?:way to|solution)',
            r'(?:really|so) (?:frustrating|annoying)',
            r'(?:wish|hope) (?:there was|it could)',
            r'(?:spent|wasted) (?:hours|days|weeks)',
        ]

    def extract_pain_signals(self, text: str) -> Dict:
        """Extract multiple pain indicators from text"""
        text_lower = text.lower()

        # Sentiment analysis
        sentiment = self.analyzer.polarity_scores(text)

        # Keyword matching
        keyword_matches = {}
        for category, keywords in self.pain_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_matches[category] = matches

        # Pattern matching
        pattern_matches = sum(1 for pattern in self.pain_patterns
                              if re.search(pattern, text_lower))

        # Question detection (questions often indicate problems)
        question_marks = text.count('?')

        # Urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'critical', 'emergency']
        urgency_score = sum(1 for word in urgency_words if word in text_lower)

        return {
            'sentiment': sentiment,
            'keyword_matches': keyword_matches,
            'pattern_matches': pattern_matches,
            'question_marks': question_marks,
            'urgency_score': urgency_score,
            'total_keyword_matches': sum(keyword_matches.values())
        }

    def calculate_pain_score(self, post_data: Dict, comments: List[Dict]) -> float:
        """Calculate comprehensive pain score"""
        # Combine post title and body
        post_text = f"{post_data.get('title', '')} {post_data.get('selftext', '')}"

        # Get pain signals from post
        post_signals = self.extract_pain_signals(post_text)

        # Get pain signals from comments
        comment_signals = []
        for comment in comments[:10]:  # Limit to first 10 comments
            comment_text = comment.get('body', '')
            if comment_text:
                comment_signals.append(self.extract_pain_signals(comment_text))

        # Calculate weighted pain score
        post_pain = self._calculate_text_pain_score(post_signals)

        if comment_signals:
            comment_pain = np.mean([self._calculate_text_pain_score(signals)
                                    for signals in comment_signals])
        else:
            comment_pain = 0

        # Post gets more weight than comments
        overall_pain = post_pain * 0.7 + comment_pain * 0.3

        return min(overall_pain, 1.0)

    def _calculate_text_pain_score(self, signals: Dict) -> float:
        """Convert pain signals to 0-1 score"""
        # Negative sentiment contributes to pain
        sentiment_pain = max(0, -signals['sentiment']['compound'])

        # Keyword matches (normalized)
        keyword_pain = min(signals['total_keyword_matches'] / 5, 1.0)

        # Pattern matches
        pattern_pain = min(signals['pattern_matches'] / 3, 1.0)

        # Questions indicate problems
        question_pain = min(signals['question_marks'] / 3, 1.0)

        # Urgency
        urgency_pain = min(signals['urgency_score'] / 2, 1.0)

        # Weighted combination
        total_pain = (
                sentiment_pain * 0.3 +
                keyword_pain * 0.3 +
                pattern_pain * 0.2 +
                question_pain * 0.1 +
                urgency_pain * 0.1
        )

        return total_pain


class NoiseDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=0.3, min_samples=5)

    def detect_noise_posts(self, embeddings: np.ndarray,
                           quality_scores: List[float]) -> List[bool]:
        """Detect posts that don't cluster well (likely noise)"""

        # Combine embeddings with quality scores
        features = np.column_stack([embeddings, quality_scores])
        features_scaled = self.scaler.fit_transform(features)

        # Cluster posts
        cluster_labels = self.clustering_model.fit_predict(features_scaled)

        # Posts labeled as -1 are noise (don't fit in any cluster)
        is_noise = cluster_labels == -1

        return is_noise.tolist()


class PainPointAnalyzer:
    def __init__(self):
        self.llm = OpenAI()
        self.patterns = {
            'workflow': ['tedious', 'manual', 'repetitive'],
            'compatibility': ['doesn\'t work with', 'incompatible'],
            'learning': ['hard to learn', 'steep curve'],
            'performance': ['slow', 'laggy', 'freezes']
        }

    def categorize_pain(self, text):
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": f"Categorize this pain point: {text}. Return JSON with keys: category, severity(1-5), opportunity_score(1-10)"
            }]
        )
        return json.loads(response.choices[0].message.content)

    def find_common_pains(self, posts):
        pain_counter = defaultdict(int)
        for post in posts:
            for point in post['pain_points']:
                category = self.categorize_pain(point)['category']
                pain_counter[category] += post['metrics']['overall']  # Weight by quality

        return sorted(pain_counter.items(), key=lambda x: x[1], reverse=True)[:5]


class ProductIdeaGenerator:
    def __init__(self):
        self.llm = OpenAI()

    def generate_ideas(self, pain_points):
        prompt = f"""Analyze these developer pain points and generate 3 product ideas:

        Pain Points:
        {pain_points}

        For each idea include:
        - Name
        - 1-sentence description
        - Target audience
        - Key differentiation
        - Estimated development difficulty (1-5)
        Format as JSON list"""

        response = self.llm.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)['ideas']


class RedditProcessor:
    def __init__(self):
        self.scorer = CompositeQualityScorer()
        self.pain_detector = AdvancedPainDetector()
        self.noise_detector = NoiseDetector()
        self.pain_analyzer = PainPointAnalyzer()
        self.idea_generator = ProductIdeaGenerator()
        self.quality_threshold = 0.65  # Only process high-quality discussions
        self.pain_threshold = 0.5  # Only include posts with significant pain signals

    def process_post(self, post):
        """Enhanced post processing with pain detection and quality scoring"""
        # Calculate quality metrics
        quality_metrics = self.scorer.score_post(post, post['comments'])

        # Calculate pain score
        pain_score = self.pain_detector.calculate_pain_score(post, post['comments'])

        # Filter by quality and pain thresholds
        if quality_metrics.overall_quality < self.quality_threshold:
            return None

        if pain_score < self.pain_threshold:
            return None

        # Extract pain points with enhanced detection
        pain_points = self.extract_enhanced_pain_points(post['comments'])

        return {
            'id': post['id'],
            'content': f"{post['title']}\n\n{post['selftext']}",
            'comments': [c['body'] for c in post['comments']],
            'metrics': {
                'engagement': quality_metrics.engagement_score,
                'depth': quality_metrics.discussion_depth,
                'sentiment': quality_metrics.response_quality,
                'patterns': quality_metrics.time_pattern_score,
                'overall': quality_metrics.overall_quality,
                'pain_score': pain_score
            },
            'pain_points': pain_points,
            'pain_categories': self._categorize_pain_points(pain_points)
        }

    def extract_enhanced_pain_points(self, comments):
        """Enhanced pain point extraction using the advanced detector"""
        pain_points = []

        for comment in comments:
            comment_text = comment.get('body', '')
            if not comment_text:
                continue

            # Get pain signals for this comment
            signals = self.pain_detector.extract_pain_signals(comment_text)
            pain_score = self.pain_detector._calculate_text_pain_score(signals)

            # Only include comments with significant pain signals
            if pain_score > 0.3:
                pain_points.append({
                    'text': comment_text,
                    'pain_score': pain_score,
                    'signals': signals
                })

        return pain_points

    def _categorize_pain_points(self, pain_points):
        """Categorize pain points for better organization"""
        categories = defaultdict(list)

        for point in pain_points:
            # Find the dominant pain category
            signals = point['signals']
            keyword_matches = signals['keyword_matches']

            if not any(keyword_matches.values()):
                categories['general'].append(point)
                continue

            # Find category with most matches
            dominant_category = max(keyword_matches.items(), key=lambda x: x[1])
            categories[dominant_category[0]].append(point)

        return dict(categories)

    def filter_noise_posts(self, posts, embeddings):
        """Filter out noise posts using clustering"""
        if not posts or not embeddings:
            return posts

        quality_scores = [p['metrics']['overall'] for p in posts]
        noise_flags = self.noise_detector.detect_noise_posts(embeddings, quality_scores)

        return [post for post, is_noise in zip(posts, noise_flags) if not is_noise]

    def generate_insights(self, processed_posts):
        """Generate product insights from processed posts"""
        if not processed_posts:
            return {"error": "No posts to analyze"}

        # Find common pain patterns
        common_pains = self.pain_analyzer.find_common_pains(processed_posts)

        # Generate product ideas
        pain_summary = [f"{category}: {score:.2f}" for category, score in common_pains]
        product_ideas = self.idea_generator.generate_ideas(pain_summary)

        return {
            'total_posts_analyzed': len(processed_posts),
            'common_pain_points': common_pains,
            'product_ideas': product_ideas,
            'average_pain_score': np.mean([p['metrics']['pain_score'] for p in processed_posts]),
            'average_quality_score': np.mean([p['metrics']['overall'] for p in processed_posts])
        }


# Qdrant Collection Setup with Quality Indexes
def setup_qdrant_collection(client: QdrantClient):
    """Setup Qdrant collection with proper indexes"""
    client.create_collection(
        collection_name="product_ideas",
        vectors_config=models.VectorParams(
            size=1536,  # OpenAI embedding size
            distance=models.Distance.COSINE
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=2000
        )
    )

    # Create payload indexes for efficient filtering
    client.create_payload_index(
        collection_name="product_ideas",
        field_name="metrics.overall",
        field_schema=models.PayloadSchemaType.FLOAT
    )

    client.create_payload_index(
        collection_name="product_ideas",
        field_name="metrics.pain_score",
        field_schema=models.PayloadSchemaType.FLOAT
    )

    client.create_payload_index(
        collection_name="product_ideas",
        field_name="pain_points",
        field_schema=models.PayloadSchemaType.KEYWORD
    )


# Example usage
def main():
    # Initialize the processor
    processor = RedditProcessor()

    # Process posts (assuming you have Reddit data)
    processed_posts = []
    embeddings = []

    # Example post processing loop
    for post in reddit_posts:  # Your Reddit posts here
        processed_post = processor.process_post(post)
        if processed_post:
            processed_posts.append(processed_post)
            # Generate embeddings for noise detection
            # embeddings.append(get_embedding(processed_post['content']))

    # Filter noise if embeddings are available
    if embeddings:
        processed_posts = processor.filter_noise_posts(processed_posts, np.array(embeddings))

    # Generate insights
    insights = processor.generate_insights(processed_posts)

    return insights


if __name__ == "__main__":
    insights = main()
    print(json.dumps(insights, indent=2))