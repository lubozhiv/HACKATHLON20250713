# ClusteringBasedNoiseDetection.py

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import List, Optional
from langchain.schema import Document


class NoiseDetector:
    def __init__(self, eps: float = 0.3, min_samples: int = 5):
        """
        Initialize the noise detector with DBSCAN parameters.

        Args:
            eps: The maximum distance between two samples for one to be considered
                 as in the neighborhood of the other.
            min_samples: The number of samples in a neighborhood for a point to be
                        considered as a core point.
        """
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=eps, min_samples=min_samples)

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

    def filter_noisy_documents(self,
                               documents: List[Document],
                               embeddings: np.ndarray,
                               remove_noise: bool = True) -> List[Document]:
        """
        Filter documents based on clustering noise detection.

        Args:
            documents: List of LangChain Document objects
            embeddings: Corresponding document embeddings
            remove_noise: If True, removes noisy documents; if False, keeps only noisy documents

        Returns:
            Filtered list of documents
        """
        if not documents or len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings must be non-empty and of equal length")

        # Extract quality scores from documents
        quality_scores = []
        for doc in documents:
            if 'quality_metrics' in doc.metadata:
                quality_scores.append(doc.metadata['quality_metrics']['overall_quality'])
            else:
                # Default score if no quality metrics available
                quality_scores.append(0.5)

        # Detect noise
        is_noise = self.detect_noise_posts(embeddings, quality_scores)

        # Filter documents
        filtered_docs = []
        for doc, noise_flag in zip(documents, is_noise):
            # Update metadata with noise detection result
            doc.metadata['noise_detection'] = {
                'is_noise': noise_flag,
                'cluster_params': {
                    'eps': self.clustering_model.eps,
                    'min_samples': self.clustering_model.min_samples
                }
            }

            # Apply filtering based on remove_noise parameter
            if (remove_noise and not noise_flag) or (not remove_noise and noise_flag):
                filtered_docs.append(doc)

        return filtered_docs