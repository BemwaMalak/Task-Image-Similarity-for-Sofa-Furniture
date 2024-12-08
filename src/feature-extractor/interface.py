from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class FeatureExtractor(ABC):
    """Abstract base class for feature extractors."""
    
    @abstractmethod
    def extract_features(self, image: np.ndarray) -> Tuple[list, Optional[np.ndarray]]:
        """
        Extract features from an image.
        
        Args:
            image (np.ndarray): Input image as a numpy array (BGR format)
            
        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: List of keypoint objects
                - descriptors: numpy array of descriptors or None if no features found
        """
        pass
    
    @abstractmethod
    def compute_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """
        Compute similarity between two sets of descriptors.
        
        Args:
            desc1 (np.ndarray): First set of descriptors
            desc2 (np.ndarray): Second set of descriptors
            
        Returns:
            float: Similarity score between 0 and 1
        """
        pass