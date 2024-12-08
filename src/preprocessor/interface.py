from abc import ABC, abstractmethod

import numpy as np


class ImagePreprocessor(ABC):
    """Abstract base class for image preprocessing operations."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image.

        Args:
            image (np.ndarray): Input image in BGR format (OpenCV default)

        Returns:
            np.ndarray: Processed image

        Raises:
            ValueError: If image is None or empty
            Exception: If critical processing error occurs
        """
        pass
