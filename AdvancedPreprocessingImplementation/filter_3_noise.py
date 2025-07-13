import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance
from sklearn.cluster import DBSCAN
from typing import List, Union, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler  # For scaling numerical features


class NoiseDetector:
    """
    A class to detect noise points in a Qdrant collection using DBSCAN
    and update the Qdrant payload with a 'noise_detection' flag.
    This version uses specified payload fields for clustering, including nested ones.
    """

    def __init__(
            self,
            qdrant_host: str,
            qdrant_port: int,
            collection_name: str,
            eps: float,
            min_samples: int,
            feature_payload_keys: List[Union[str, List[str]]]  # Can now be nested list for nested keys
    ):
        """
        Initializes the NoiseDetector with Qdrant and DBSCAN parameters.

        Args:
            qdrant_host (str): Host for Qdrant client.
            qdrant_port (int): Port for Qdrant client.
            collection_name (str): Name of the Qdrant collection to operate on.
            eps (float): The maximum distance between two samples for one to be considered
                         as in the neighborhood of the other (DBSCAN parameter).
            min_samples (int): The number of samples (or total weight) in a neighborhood
                               for a point to be considered as a core point (DBSCAN parameter).
            feature_payload_keys (List[Union[str, List[str]]]): A list of payload keys
                                              whose values will be used as features for DBSCAN.
                                              Can be a string for top-level keys (e.g., "score")
                                              or a list of strings for nested keys (e.g., ["pain_metrics", "pain_score"]).
                                              All extracted values must be numerical.
        """
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_metric = 'euclidean'  # Standard for numerical features
        self.feature_payload_keys = feature_payload_keys

    def _get_nested_value(self, data: Dict[str, Any], keys: Union[str, List[str]], default: Any = 0.0) -> Any:
        """
        Safely retrieves a nested value from a dictionary using a list of keys.
        Returns default if any key in the path is not found or if the final value is not numerical.
        """
        if isinstance(keys, str):
            keys = [keys]  # Convert single key string to list for uniform processing

        current_value = data
        for key in keys:
            if isinstance(current_value, dict) and key in current_value:
                current_value = current_value[key]
            else:
                return default  # Key not found at this level

        # Ensure the final extracted value is numerical
        if isinstance(current_value, (int, float)):
            return current_value
        else:
            print(
                f"Warning: Extracted value for path {keys} is not numerical ({current_value}). Using default {default}.")
            return default

    def _get_qdrant_data(self) -> Tuple[List[Union[int, str]], np.ndarray, List[Dict[str, Any]]]:
        """
        Retrieves document IDs, and features extracted from payloads from the Qdrant collection.

        Returns:
            tuple: A tuple containing:
                   - List of document IDs.
                   - NumPy array of features extracted from payloads (scaled).
                   - List of full document payloads.
        """
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' exists with {collection_info.points_count} points.")
        except Exception as e:
            print(f"Error accessing collection '{self.collection_name}': {e}")
            print("Please ensure the collection exists.")
            return [], np.array([]), []

        scroll_result, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,  # Adjust limit as needed, or use pagination
            with_payload=True,
            with_vectors=False  # Not using vectors for DBSCAN in this modified version
        )

        document_ids = [point.id for point in scroll_result]
        document_payloads = [point.payload for point in scroll_result]

        feature_data = []
        for payload in document_payloads:
            row_features = []
            for key_path in self.feature_payload_keys:
                # Use the new helper method to get nested values
                value = self._get_nested_value(payload, key_path, default=0.0)
                row_features.append(value)
            feature_data.append(row_features)

        features_array = np.array(feature_data)

        # Scale numerical features if they have different ranges.
        if features_array.size > 0 and features_array.shape[1] > 0:
            scaler = StandardScaler()
            features_array_scaled = scaler.fit_transform(features_array)
        else:
            features_array_scaled = features_array  # No scaling if no data or no features

        return document_ids, features_array_scaled, document_payloads

    def detect_noise(self) -> Tuple[List[Union[int, str]], np.ndarray, List[Dict[str, Any]], np.ndarray]:
        """
        Performs DBSCAN clustering to identify noise points using specified payload features.

        Returns:
            tuple: A tuple containing:
                   - List of all document IDs.
                   - NumPy array of all document features (scaled).
                   - List of all document payloads.
                   - NumPy array of cluster labels (-1 for noise).
        """
        doc_ids, features, doc_payloads = self._get_qdrant_data()

        if not doc_ids or features.size == 0 or features.shape[1] == 0:
            print("No documents or features retrieved. Cannot perform noise detection.")
            return [], np.array([]), [], np.array([])

        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.dbscan_metric)
        cluster_labels = dbscan.fit_predict(features)

        noise_point_indices = np.where(cluster_labels == -1)[0]
        print(f"Found {len(noise_point_indices)} noise points.")

        return doc_ids, features, doc_payloads, cluster_labels

    def update_noise_flags_in_qdrant(
            self,
            noise_document_ids: List[Union[int, str]],
            all_document_ids: List[Union[int, str]]
    ):
        """
        Updates the 'noise_detection' payload flag in Qdrant for all documents.
        True for noise points, False for non-noise points.
        """

        noise_ids_set = set(noise_document_ids)

        noise_ids_for_qdrant = [doc_id for doc_id in all_document_ids if doc_id in noise_ids_set]
        non_noise_ids_for_qdrant = [doc_id for doc_id in all_document_ids if doc_id not in noise_ids_set]

        if noise_ids_for_qdrant:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"noise_detection": True},
                points=noise_ids_for_qdrant
            )
            print(f"Set 'noise_detection': True for {len(noise_ids_for_qdrant)} noise points.")
        else:
            print("No noise points to mark as True.")

        if non_noise_ids_for_qdrant:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"noise_detection": False},
                points=non_noise_ids_for_qdrant
            )
        else:
            print("No non-noise points to mark as False.")


if __name__ == "__main__":
    # --- Configuration ---
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    COLLECTION_NAME = "reddit_pain_documents"

    # DBSCAN Parameters - These are CRITICAL and must be manually set.
    # Adjust EPS and MIN_SAMPLES based on the distribution and scale of your pain_score/other features.
    EPS = 0.5  # Placeholder: Adjust this based on your data's density in the feature space
    MIN_SAMPLES = 5  # Placeholder: Adjust this based on expected cluster size

    FEATURE_PAYLOAD_KEYS = [
        ["pain_metrics", "pain_score"],
        "score",  # Top-level score
        "upvote_ratio",  # Top-level upvote_ratio
        "num_comments",  # Top-level num_comments
        # Example of other potential nested numerical fields from your JSON:
        ["pain_metrics", "post_pain", "sentiment", "compound"],
        ["pain_metrics", "post_pain", "keyword_matches", "frustration"],
        ["pain_metrics", "post_pain", "total_keyword_matches"],
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

    # Identify noise points from the cluster labels
    noise_point_indices = np.where(cluster_labels == -1)[0]
    noise_document_ids = [all_doc_ids[i] for i in noise_point_indices]

    # Update Qdrant with the noise flags
    detector.update_noise_flags_in_qdrant(noise_document_ids, all_doc_ids)

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))

    print("Process complete. You can now query Qdrant and filter by 'noise_detection' payload field.")