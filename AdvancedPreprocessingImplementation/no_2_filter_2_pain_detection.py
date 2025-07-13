# EnhancedPainDetection.py

import re
import numpy as np
from typing import List, Dict, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from langchain.schema import Document


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

    def filter_documents_by_pain(
            self,
            documents: List[Document],
            min_pain_score: float = 0.3,
            max_pain_score: float = 1.0
    ) -> List[Document]:
        """
        Filter documents based on pain score.

        Args:
            documents: List of LangChain Document objects
            min_pain_score: Minimum pain score to keep (0.0-1.0)
            max_pain_score: Maximum pain score to keep (0.0-1.0)

        Returns:
            Filtered list of documents within the specified pain score range
        """
        filtered_docs = []

        for doc in documents:
            # Convert document metadata to post_data format
            post_data = doc.metadata
            comments = []

            # Extract comments from page_content
            page_content = doc.page_content
            if "Comments:" in page_content:
                comments_section = page_content.split("Comments:")[1]
                comments = [{"body": comment.split(": ", 1)[1]} for comment in comments_section.split("\n\n")
                            if comment.startswith("Comment ")]

            # Calculate pain score
            pain_score = self.calculate_pain_score(post_data, comments)

            # Keep if within pain score range
            if min_pain_score <= pain_score <= max_pain_score:
                # Add pain metrics to document metadata
                doc.metadata.update({
                    "pain_metrics": {
                        "pain_score": pain_score,
                        "post_pain": self.extract_pain_signals(
                            f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                        ),
                        "comments_pain": [self.extract_pain_signals(comment["body"])
                                          for comment in comments[:10]] if comments else []
                    }
                })
                filtered_docs.append(doc)

        return filtered_docs