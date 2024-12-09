from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class IFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract features from an image.

        Args:
            image (np.ndarray): Input image as a numpy array (BGR format)

        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: List of feature vectors (e.g., color histograms)
                - descriptors: Feature descriptors (e.g., texture features)

        Raises:
            ValueError: If image is None, empty, or not a valid numpy array
            TypeError: If image is not a numpy array
            Exception: If feature extraction fails for any reason
        """
        pass

    @abstractmethod
    def compute_similarity_matrix(
        self, features_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between all feature pairs.

        Args:
            features_list: List of feature tuples, each containing (feature_vector, descriptors)

        Returns:
            np.ndarray: Similarity matrix where element [i,j] is the similarity score between
                       features_list[i] and features_list[j]

        Raises:
            ValueError: If features_list is empty or contains invalid features
            TypeError: If features_list is not a list or contains invalid types
            Exception: If similarity computation fails for any reason
        """
        pass

    @abstractmethod
    def find_most_similar(
        self,
        query_features: Tuple[np.ndarray, np.ndarray],
        database_features: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[int, float]:
        """
        Find the most similar features from a database of features.

        Args:
            query_features: Tuple of (feature_vector, descriptors) for query image
            database_features: List of feature tuples for database images

        Returns:
            tuple: (index, similarity_score)
                - index: Index of most similar features in database_features
                - similarity_score: Similarity score of the best match

        Raises:
            ValueError: If database_features is empty or features are invalid
            TypeError: If query_features is not a tuple or database_features is not a list
            Exception: If similarity search fails for any reason
        """
        pass
