from typing import Protocol

import numpy as np


class IPreprocessorService(Protocol):
    """Interface for image preprocessing service."""

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image

        Raises:
            Exception: If preprocessing fails
        """
        ...
