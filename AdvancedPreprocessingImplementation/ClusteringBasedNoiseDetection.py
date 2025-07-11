# ClusteringBasedNoiseDetection.py

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np


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


    /
