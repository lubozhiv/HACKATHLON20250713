import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from sklearn.cluster import DBSCAN
from typing import List, Union, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class NoiseDetector:
    """
    A class to detect noise points in a Qdrant collection using DBSCAN
    and update the Qdrant payload with a 'noise_detection' flag.
    """

    def __init__(
            self,
            qdrant_host: str,
            qdrant_port: int,
            collection_name: str,
            eps: float,
            min_samples: int,
            feature_payload_keys: List[Union[str, List[str]]]
    ):
        """
        Initializes the NoiseDetector with Qdrant and DBSCAN parameters.
        """
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_metric = 'euclidean'
        self.feature_payload_keys = feature_payload_keys

    def _get_nested_value(self, data: Dict[str, Any], keys: Union[str, List[str]], default: Any = 0.0) -> Any:
        """
        Safely retrieves a nested value from a dictionary using a list of keys.
        """
        if isinstance(keys, str):
            keys = [keys]

        current_value = data
        for key in keys:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                logger.debug(f"Key path {keys} not found in data. Using default {default}.")
                return default

        # Ensure the final extracted value is numerical
        if isinstance(current_value, (int, float)):
            return current_value
        else:
            logger.warning(
                f"Extracted value for path {keys} is not numerical ({current_value}). Using default {default}.")
            return default

    def _get_qdrant_data(self) -> Tuple[List[Union[int, str]], np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieves document IDs and features extracted from payloads from the Qdrant collection.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points.")
        except Exception as e:
            print(f"Error accessing collection '{self.collection_name}': {e}")
            return [], np.array([]), []

        # Use pagination to handle large collections
        all_points = []
        offset = None

        while True:
            try:
                scroll_result, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,  # Process in smaller batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not scroll_result:
                    break

                all_points.extend(scroll_result)

                if next_offset is None:
                    break

                offset = next_offset

            except Exception as e:
                print(f"Error during scroll operation: {e}")
                break

        if not all_points:
            print("No points retrieved from collection.")
            return [], np.array([]), []

        document_ids = [point.id for point in all_points]
        document_payloads = [point.payload for point in all_points]

        # Extract features with better error handling
        feature_data = []
        valid_indices = []

        for idx, payload in enumerate(document_payloads):
            row_features = []
            valid_row = True

            for key_path in self.feature_payload_keys:
                value = self._get_nested_value(payload, key_path, default=0.0)

                # Additional validation
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    logger.warning(
                        f"Invalid value {value} for key path {key_path} in document {document_ids[idx]}. Using 0.0.")
                    value = 0.0

                row_features.append(value)

            # Only include rows where all features are valid
            if all(isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val) for val in row_features):
                feature_data.append(row_features)
                valid_indices.append(idx)
            else:
                logger.warning(f"Skipping document {document_ids[idx]} due to invalid features.")

        if not feature_data:
            print("No valid feature data extracted.")
            return [], np.array([]), []

        # Filter document_ids and payloads to match valid feature data
        valid_document_ids = [document_ids[i] for i in valid_indices]
        valid_document_payloads = [document_payloads[i] for i in valid_indices]

        features_array = np.array(feature_data)
        print(f"Extracted features shape: {features_array.shape}")
        print(f"Feature statistics:\n{np.array([np.mean(features_array, axis=0), np.std(features_array, axis=0)])}")

        # Scale features
        if features_array.size > 0 and features_array.shape[1] > 0:
            scaler = StandardScaler()
            features_array_scaled = scaler.fit_transform(features_array)
            print(f"Features scaled. New shape: {features_array_scaled.shape}")
        else:
            features_array_scaled = features_array

        return valid_document_ids, features_array_scaled, valid_document_payloads

    def detect_noise(self) -> Tuple[List[Union[int, str]], np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Performs DBSCAN clustering to identify noise points using specified payload features.
        """
        doc_ids, features, doc_payloads = self._get_qdrant_data()

        if not doc_ids or features.size == 0:
            print("No documents or features retrieved. Cannot perform noise detection.")
            return [], np.array([]), [], np.array([])

        if features.shape[1] == 0:
            print("No valid features extracted. Cannot perform noise detection.")
            return [], np.array([]), [], np.array([])

        print(f"Performing DBSCAN with eps={self.eps}, min_samples={self.min_samples}")
        print(f"Input data shape: {features.shape}")

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.dbscan_metric)
        cluster_labels = dbscan.fit_predict(features)

        noise_point_indices = np.where(cluster_labels == -1)[0]
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        print(f"DBSCAN Results:")
        print(f"  Total points: {len(cluster_labels)}")
        print(f"  Noise points: {len(noise_point_indices)}")
        print(f"  Clusters found: {len(unique_labels) - (1 if -1 in unique_labels else 0)}")
        print(f"  Cluster distribution: {dict(zip(unique_labels, counts))}")

        return doc_ids, features, doc_payloads, cluster_labels

    def update_noise_flags_in_qdrant(
            self,
            noise_document_ids: List[Union[int, str]],
            all_document_ids: List[Union[int, str]]
    ):
        """
        Updates the 'noise_detection' payload flag in Qdrant for all documents.
        """
        noise_ids_set = set(noise_document_ids)

        # Process in batches to avoid memory issues
        batch_size = 100

        # Update noise points
        noise_ids_list = [doc_id for doc_id in all_document_ids if doc_id in noise_ids_set]
        for i in range(0, len(noise_ids_list), batch_size):
            batch_ids = noise_ids_list[i:i + batch_size]
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"noise_detection": True},
                    points=batch_ids
                )
            except Exception as e:
                print(f"Error setting noise_detection=True for batch {i // batch_size + 1}: {e}")

        # Update non-noise points
        non_noise_ids_list = [doc_id for doc_id in all_document_ids if doc_id not in noise_ids_set]
        for i in range(0, len(non_noise_ids_list), batch_size):
            batch_ids = non_noise_ids_list[i:i + batch_size]
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"noise_detection": False},
                    points=batch_ids
                )
            except Exception as e:
                print(f"Error setting noise_detection=False for batch {i // batch_size + 1}: {e}")

        print(f"Updated noise flags: {len(noise_ids_list)} noise points, {len(non_noise_ids_list)} non-noise points")

    def get_noise_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about noise detection results from the collection.
        """
        try:
            # Count noise and non-noise points
            noise_points = self.client.count(
                collection_name=self.collection_name,
                count_filter={
                    "must": [{"key": "noise_detection", "match": {"value": True}}]
                }
            )

            non_noise_points = self.client.count(
                collection_name=self.collection_name,
                count_filter={
                    "must": [{"key": "noise_detection", "match": {"value": False}}]
                }
            )

            total_points = self.client.count(collection_name=self.collection_name)

            return {
                "total_points": total_points.count,
                "noise_points": noise_points.count,
                "non_noise_points": non_noise_points.count,
                "noise_ratio": noise_points.count / total_points.count if total_points.count > 0 else 0
            }

        except Exception as e:
            print(f"Error getting noise statistics: {e}")
            return {}


if __name__ == "__main__":
    # Configuration
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "reddit_pain_documents"
    EPS = 0.5
    MIN_SAMPLES = 5

    FEATURE_PAYLOAD_KEYS = [
        ["pain_metrics", "pain_score"],
        "score",
        "upvote_ratio",
        "num_comments",
        ["pain_metrics", "post_pain", "sentiment", "compound"],
        ["pain_metrics", "post_pain", "keyword_matches", "frustration"],
        ["pain_metrics", "post_pain", "total_keyword_matches"],
        ["quality_metrics", "overall_quality"],
        ["quality_metrics", "engagement_score"]
    ]

    detector = NoiseDetector(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection_name=COLLECTION_NAME,
        eps=EPS,
        min_samples=MIN_SAMPLES,
        feature_payload_keys=FEATURE_PAYLOAD_KEYS
    )

    # Perform noise detection
    all_doc_ids, all_features, all_doc_payloads, cluster_labels = detector.detect_noise()

    if not all_doc_ids:
        print("Exiting due to no data.")
        exit()

    # Identify and update noise points
    noise_point_indices = np.where(cluster_labels == -1)[0]
    noise_document_ids = [all_doc_ids[i] for i in noise_point_indices]
    detector.update_noise_flags_in_qdrant(noise_document_ids, all_doc_ids)

    # Print final statistics
    stats = detector.get_noise_statistics()
    print(f"\nFinal Statistics: {stats}")
    print("Process complete.")