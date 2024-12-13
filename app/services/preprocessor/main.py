import numpy as np

from app.config import GRABCUT_ITERATIONS, MAX_IMAGE_SIZE, PADDING
from src.preprocessor import SofaSegmenter

from .interface import IPreprocessorService


class PreprocessorService(IPreprocessorService):
    def __init__(self):
        """Initialize the preprocessor service with SofaSegmenter."""
        self.preprocessor = SofaSegmenter(
            padding=PADDING, max_size=MAX_IMAGE_SIZE, iterations=GRABCUT_ITERATIONS
        )

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image using SofaSegmenter.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image

        Raises:
            Exception: If preprocessing fails
        """
        try:
            processed_image = self.preprocessor.preprocess(image)
            return processed_image

        except Exception as e:
            raise Exception(f"Failed to preprocess image: {str(e)}")
