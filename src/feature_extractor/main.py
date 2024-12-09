from typing import List, Tuple

import cv2
import numpy as np

from .interface import IFeatureExtractor


class ColorHistogramExtractor(IFeatureExtractor):
    """Color Histogram and Texture feature extractor implementation."""

    def __init__(self, hist_bins=32):
        """
        Initialize the Color Histogram extractor.

        Args:
            hist_bins (int): Number of bins for the color histogram

        Raises:
            ValueError: If hist_bins is less than or equal to 0
        """
        if not isinstance(hist_bins, int) or hist_bins <= 0:
            raise ValueError("hist_bins must be a positive integer")
        self.hist_bins = hist_bins

    def extract_features(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract color histogram and texture features from an image.

        Args:
            image (np.ndarray): Input image as a numpy array (BGR format)

        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: Color histogram features
                - descriptors: Texture features

        Raises:
            ValueError: If image is None, empty, or not a valid numpy array
            TypeError: If image is not a numpy array
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        if image is None or image.size == 0 or len(image.shape) != 3:
            raise ValueError("Invalid input image: must be a non-empty 3-channel image")
        if image.dtype != np.uint8:
            raise ValueError("Image must be in uint8 format")

        try:
            # Convert BGR to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        except cv2.error as e:
            raise ValueError(f"Failed to convert image to HSV: {str(e)}")

        try:
            # Calculate color histogram for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])

            # Normalize histograms
            cv2.normalize(hist_h, hist_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_s, hist_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_v, hist_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Combine histograms
            color_features = np.concatenate([hist_h, hist_s, hist_v])
        except Exception as e:
            raise ValueError(f"Failed to compute color histogram: {str(e)}")

        try:
            # Calculate texture features using gray-scale image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize image to a fixed size for consistent texture features
            gray_resized = cv2.resize(gray, (256, 256))
            
            # Calculate GLCM features
            glcm = self._compute_glcm(gray_resized)
            texture_features = self._compute_glcm_features(glcm)
            
            # Reshape texture features to column vector
            texture_features = texture_features.reshape(-1, 1)
            
        except Exception as e:
            raise ValueError(f"Failed to compute texture features: {str(e)}")

        return color_features, texture_features

    def _compute_glcm(self, gray_image: np.ndarray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        """
        Compute Gray Level Co-occurrence Matrix.
        
        Args:
            gray_image: Grayscale image
            distances: List of pixel pair distances
            angles: List of pixel pair angles
            
        Returns:
            GLCM matrix
        """
        if not isinstance(gray_image, np.ndarray):
            raise TypeError("gray_image must be a numpy array")
        if gray_image.ndim != 2:
            raise ValueError("gray_image must be a 2D array")
            
        # Normalize gray levels to reduce GLCM size
        levels = 32
        gray_normalized = (gray_image * (levels - 1) / 255).astype(np.uint8)
        
        # Calculate GLCM for each distance and angle
        glcm = np.zeros((levels, levels))
        for distance in distances:
            for angle in angles:
                offset_x = int(distance * np.cos(angle))
                offset_y = int(distance * np.sin(angle))
                
                # Calculate co-occurrence matrix
                rows, cols = gray_normalized.shape
                for i in range(rows):
                    for j in range(cols):
                        i2 = i + offset_y
                        j2 = j + offset_x
                        if 0 <= i2 < rows and 0 <= j2 < cols:
                            glcm[gray_normalized[i, j], gray_normalized[i2, j2]] += 1
                            
        # Normalize GLCM
        if glcm.sum() > 0:
            glcm = glcm / glcm.sum()
            
        return glcm

    def _compute_glcm_features(self, glcm: np.ndarray) -> np.ndarray:
        """
        Compute features from GLCM.
        
        Args:
            glcm: Gray Level Co-occurrence Matrix
            
        Returns:
            Array of GLCM features
        """
        if not isinstance(glcm, np.ndarray):
            raise TypeError("GLCM must be a numpy array")
        if glcm.ndim != 2 or glcm.shape[0] != glcm.shape[1]:
            raise ValueError("GLCM must be a square 2D array")
            
        # Create meshgrid for efficient computation
        rows, cols = glcm.shape
        i, j = np.meshgrid(range(rows), range(cols), indexing='ij')
        
        # Calculate features
        contrast = np.sum(glcm * ((i - j) ** 2))
        dissimilarity = np.sum(glcm * np.abs(i - j))
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        energy = np.sum(glcm ** 2)
        correlation = np.sum(glcm * (i - np.mean(i)) * (j - np.mean(j))) / (np.std(i) * np.std(j))
        
        features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
        return features

    def compute_similarity_matrix(
        self, features_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix between all feature pairs.

        Args:
            features_list: List of feature tuples, each containing (color_hist, texture_features)

        Returns:
            np.ndarray: Similarity matrix

        Raises:
            ValueError: If features_list is empty or contains invalid features
            TypeError: If features_list is not a list
        """
        if not isinstance(features_list, list):
            raise TypeError("features_list must be a list")
        if not features_list:
            raise ValueError("features_list cannot be empty")

        try:
            n = len(features_list)
            similarity_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                        continue

                    score = self._compute_pair_similarity(
                        features_list[i], features_list[j]
                    )
                    similarity_matrix[i, j] = score
                    similarity_matrix[j, i] = score

            return similarity_matrix
        except Exception as e:
            raise ValueError(f"Failed to compute similarity matrix: {str(e)}")

    def _compute_pair_similarity(
        self,
        features1: Tuple[np.ndarray, np.ndarray],
        features2: Tuple[np.ndarray, np.ndarray],
    ) -> float:
        """
        Compute similarity between two feature sets.

        Args:
            features1: Tuple of (color_hist1, texture_features1)
            features2: Tuple of (color_hist2, texture_features2)

        Returns:
            float: Similarity score between 0 and 1

        Raises:
            ValueError: If features are invalid or incompatible
            TypeError: If features are not tuples
        """
        if not isinstance(features1, tuple) or not isinstance(features2, tuple):
            raise TypeError("Features must be tuples")
        if len(features1) != 2 or len(features2) != 2:
            raise ValueError("Features must contain exactly 2 elements")

        try:
            color_hist1, texture_features1 = features1
            color_hist2, texture_features2 = features2
            
            if not isinstance(color_hist1, np.ndarray) or not isinstance(
                color_hist2, np.ndarray
            ):
                raise TypeError("Color histograms must be numpy arrays")
            if not isinstance(texture_features1, np.ndarray) or not isinstance(
                texture_features2, np.ndarray
            ):
                raise TypeError("Texture features must be numpy arrays")
            if color_hist1.shape != color_hist2.shape:
                raise ValueError("Color histograms must have the same shape")
            if texture_features1.shape != texture_features2.shape:
                raise ValueError("Texture features must have the same shape")

            # Compare color histograms using correlation
            color_similarity = cv2.compareHist(
                color_hist1, color_hist2, cv2.HISTCMP_CORREL
            )

            # Compare texture features using cosine similarity
            texture_similarity = self._cosine_similarity(
                texture_features1.flatten(), texture_features2.flatten()
            )

            # Combine similarities
            combined_similarity = 0.7 * color_similarity + 0.3 * texture_similarity

            return max(0.0, min(1.0, combined_similarity))
        except Exception as e:
            raise ValueError(f"Failed to compute pair similarity: {str(e)}")

    def find_most_similar(
        self,
        query_features: Tuple[np.ndarray, np.ndarray],
        database_features: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[int, float]:
        """
        Find the most similar features from a database of features.

        Args:
            query_features: Tuple of (color_hist, texture_features) for query image
            database_features: List of feature tuples for database images

        Returns:
            tuple: (index, similarity_score)
                - index: Index of most similar features in database_features
                - similarity_score: Similarity score of the best match

        Raises:
            ValueError: If database_features is empty or features are invalid
            TypeError: If inputs are of incorrect type
        """
        if not isinstance(query_features, tuple):
            raise TypeError("query_features must be a tuple")
        if not isinstance(database_features, list):
            raise TypeError("database_features must be a list")
        if not database_features:
            raise ValueError("database_features cannot be empty")

        try:
            similarities = [
                self._compute_pair_similarity(query_features, db_features)
                for db_features in database_features
            ]
            best_idx = int(np.argmax(similarities))
            return best_idx, float(similarities[best_idx])
        except Exception as e:
            raise ValueError(f"Failed to find most similar features: {str(e)}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            float: Cosine similarity

        Raises:
            ValueError: If vectors have different shapes or are empty
            TypeError: If inputs are not numpy arrays
        """
        if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
            raise TypeError("Vectors must be numpy arrays")
        if v1.shape != v2.shape:
            raise ValueError("Vectors must have the same shape")
        if v1.size == 0:
            raise ValueError("Vectors cannot be empty")

        try:
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm_product == 0:
                return 0.0
            return np.dot(v1, v2) / norm_product
        except Exception as e:
            raise ValueError(f"Failed to compute cosine similarity: {str(e)}")
