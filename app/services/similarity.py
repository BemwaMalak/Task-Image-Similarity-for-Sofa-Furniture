import os
from typing import Tuple

import numpy as np

from app.config import HIST_BINS, ROOT_DIR
from src.db.models import Product
from src.db.repos import ProductRepository
from src.feature_extractor import ColorHistogramExtractor


class SimilarityService:
    def __init__(self, product_repo: ProductRepository):
        """
        Initialize the similarity service.

        Args:
            product_repo: Repository for product database operations
        """
        self.feature_extractor = ColorHistogramExtractor(hist_bins=HIST_BINS)
        self.product_repo = product_repo

    def extract_features(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        try:
            # Extract features
            features = self.feature_extractor.extract_features(image)
            return features

        except Exception as e:
            raise Exception(f"Failed to extract features: {str(e)}")

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
        try:
            # Get all products
            products = self.product_repo.get_all()
            if not products:
                return None, 0.0

            # Load features for all products
            product_features = []
            valid_products = []
            for product in products:
                if product.features_file_path is not None and os.path.exists(
                    os.path.join(ROOT_DIR, str(product.features_file_path))
                ):
                    try:
                        # Load features from npz file
                        data = np.load(
                            os.path.join(ROOT_DIR, str(product.features_file_path)),
                            allow_pickle=True,
                        )
                        features = (
                            np.array(data["keypoints"]),
                            np.array(data["descriptors"]),
                        )
                        product_features.append(features)
                        valid_products.append(product)
                    except Exception:
                        continue

            if not product_features:
                return None, 0.0

            # Use feature extractor to find most similar product
            best_idx, similarity = self.feature_extractor.find_most_similar(
                query_features, product_features
            )

            return valid_products[best_idx], similarity

        except Exception as e:
            raise Exception(f"Failed to find similar product: {str(e)}")
