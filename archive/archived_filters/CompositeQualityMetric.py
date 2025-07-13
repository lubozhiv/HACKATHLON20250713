# CompositeQualityMetric.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

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