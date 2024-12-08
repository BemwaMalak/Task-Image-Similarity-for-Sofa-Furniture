from interface import FeatureExtractor
from typing import Tuple, List
import cv2
import numpy as np

class ColorHistogramExtractor(FeatureExtractor):
    """Color Histogram and Texture feature extractor implementation."""
    
    def __init__(self, hist_bins=32):
        """
        Initialize the Color Histogram extractor.
        
        Args:
            hist_bins (int): Number of bins for the color histogram
        """
        self.hist_bins = hist_bins
        
    def extract_features(self, image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Extract color histogram and texture features from an image.
        
        Args:
            image (np.ndarray): Input image as a numpy array (BGR format)
            
        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: List of color histogram features
                - descriptors: Texture features
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
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
        
        # Calculate texture features using gray-scale image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute texture features (e.g., using gradient magnitude)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_features = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return [color_features], texture_features
    
    def compute_similarity_matrix(self, features_list: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Compute pairwise similarity matrix between all feature pairs.
        
        Args:
            features_list: List of feature tuples, each containing (color_hist, texture_features)
            
        Returns:
            np.ndarray: Similarity matrix where element [i,j] is the similarity score between
                       features_list[i] and features_list[j]
        """
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i,j] = 1.0
                    continue
                    
                score = self._compute_pair_similarity(features_list[i], features_list[j])
                similarity_matrix[i,j] = score
                similarity_matrix[j,i] = score
                
        return similarity_matrix
    
    def _compute_pair_similarity(self, features1: Tuple[np.ndarray, np.ndarray],
                               features2: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        Compute similarity between two feature sets.
        
        Args:
            features1: Tuple of (color_hist1, texture_features1)
            features2: Tuple of (color_hist2, texture_features2)
            
        Returns:
            float: Similarity score between 0 and 1
        """
        color_hist1, texture_features1 = features1
        color_hist2, texture_features2 = features2
        
        # Compare color histograms using correlation
        color_similarity = cv2.compareHist(color_hist1, color_hist2, cv2.HISTCMP_CORREL)
        
        # Compare texture features using cosine similarity
        texture_similarity = self._cosine_similarity(texture_features1, texture_features2)
        
        # Combine similarities
        combined_similarity = 0.7 * color_similarity + 0.3 * texture_similarity
        
        return max(0.0, min(1.0, combined_similarity))
    
    def find_most_similar(self, query_features: Tuple[np.ndarray, np.ndarray],
                         database_features: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[int, float]:
        """
        Find the most similar features from a database of features.
        
        Args:
            query_features: Tuple of (color_hist, texture_features) for query image
            database_features: List of feature tuples for database images
            
        Returns:
            tuple: (index, similarity_score)
                - index: Index of most similar features in database_features
                - similarity_score: Similarity score of the best match
        """
        similarities = [self._compute_pair_similarity(query_features, db_features) 
                      for db_features in database_features]
        best_idx = int(np.argmax(similarities))
        return best_idx, float(similarities[best_idx])
    
    def _calculate_glcm(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Gray-Level Co-Occurrence Matrix."""
        gray_image = cv2.resize(gray_image, (64, 64))
        levels = 8
        gray_image = ((gray_image / 256) * levels).astype(np.uint8)
        
        glcm = np.zeros((levels, levels))
        h, w = gray_image.shape
        
        for i in range(h-1):
            for j in range(w-1):
                current = gray_image[i, j]
                right = gray_image[i, j+1]
                glcm[current, right] += 1
                
        # Normalize GLCM
        glcm = glcm / glcm.sum()
        return glcm
    
    def _extract_glcm_features(self, glcm: np.ndarray) -> np.ndarray:
        """Extract features from GLCM."""
        contrast = np.sum(np.square(np.arange(glcm.shape[0])) * glcm)
        homogeneity = np.sum(glcm / (1 + np.square(np.arange(glcm.shape[0]))))
        energy = np.sum(np.square(glcm))
        correlation = np.sum(glcm * np.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[0])))
        
        return np.array([contrast, homogeneity, energy, correlation])
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))