from typing import Protocol, Tuple

import numpy as np

from src.db.models import Product


class ISimilarityService(Protocol):
    """Interface for image similarity service."""

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from an image.

        Args:
            image: Input image in BGR format

        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: List of color histogram features
                - descriptors: Texture features

        Raises:
            Exception: If feature extraction fails
        """
        ...

    def find_most_similar(
        self, query_features: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[Product | None, float]:
        """
        Find the most similar product based on image features.

        Args:
            query_features: Features of the query image

        Returns:
            Tuple of (product, similarity_score)

        Raises:
            Exception: If similarity search fails
        """
        ...
