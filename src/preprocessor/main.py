import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .interface import IPreprocessor


@dataclass
class BoundingBox:
    """Dataclass to represent a bounding box."""

    x: int
    y: int
    width: int
    height: int


class SofaSegmenter(IPreprocessor):
    """Sofa segmentation and background removal using GrabCut algorithm."""

    def __init__(self, padding: int = 10, max_size: int = 800, iterations: int = 1):
        """
        Initialize the sofa segmenter.

        Args:
            padding (int): Padding to add around the segmented sofa
            max_size (int): Maximum dimension size for resizing while maintaining aspect ratio
            iterations (int): Number of GrabCut iterations

        Raises:
            ValueError: If any parameters are invalid
        """
        if padding < 0 or max_size <= 0 or iterations <= 0:
            raise ValueError(
                "Invalid parameters: padding must be >= 0, max_size and iterations must be > 0"
            )

        self.padding = padding
        self.max_size = max_size
        self.iterations = iterations
        self.logger = logging.getLogger(__name__)

    def _resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image while maintaining aspect ratio if it exceeds max_size.

        Args:
            image: Input image

        Returns:
            Tuple[np.ndarray, float]: (Resized image, scale factor)

        Raises:
            ValueError: If input image is empty or invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")

        height, width = image.shape[:2]
        max_dim = max(height, width)

        if max_dim > self.max_size:
            scale = self.max_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            try:
                resized = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_AREA
                )
                return resized, scale
            except cv2.error as e:
                self.logger.error(f"Failed to resize image: {e}")
                raise Exception(f"Image resizing failed: {e}")
        return image, 1.0

    def _create_initial_mask(self, height: int, width: int) -> np.ndarray:
        """
        Create initial mask for GrabCut with sofa-specific regions.

        Args:
            height (int): Image height
            width (int): Image width

        Returns:
            np.ndarray: Initialized mask with background/foreground regions
        """
        # Initialize as probable background
        mask = np.zeros((height, width), np.uint8) + cv2.GC_PR_BGD

        # Border parameters (5% of the smaller dimension)
        border = int(min(height, width) * 0.05)

        # Mark borders as definite background
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Mark central region as probable foreground
        center_y, center_x = height // 2, width // 2
        center_h = int(height * 0.6)  # 60% of height
        center_w = int(width * 0.6)  # 60% of width

        start_y = center_y - center_h // 2
        end_y = center_y + center_h // 2
        start_x = center_x - center_w // 2
        end_x = center_x + center_w // 2

        # Ensure coordinates are within bounds
        start_y, end_y = max(0, start_y), min(height, end_y)
        start_x, end_x = max(0, start_x), min(width, end_x)

        # Mark central region as probable foreground
        mask[start_y:end_y, start_x:end_x] = cv2.GC_PR_FGD

        # Mark inner region as definite foreground
        inner_h = int(center_h * 0.6)  # 60% of center height
        inner_w = int(center_w * 0.6)  # 60% of center width

        inner_start_y = center_y - inner_h // 2
        inner_end_y = center_y + inner_h // 2
        inner_start_x = center_x - inner_w // 2
        inner_end_x = center_x + inner_w // 2

        # Ensure coordinates are within bounds
        inner_start_y, inner_end_y = max(0, inner_start_y), min(height, inner_end_y)
        inner_start_x, inner_end_x = max(0, inner_start_x), min(width, inner_end_x)

        # Mark inner region as definite foreground
        mask[inner_start_y:inner_end_y, inner_start_x:inner_end_x] = cv2.GC_FGD

        return mask

    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates from the largest contour in the mask.

        Args:
            mask: Binary mask

        Returns:
            Tuple[int, int, int, int]: (x, y, width, height) of bounding box
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, width, height = cv2.boundingRect(largest_contour)
            return (x, y, width, height)
        return (0, 0, mask.shape[1], mask.shape[0])

    def _segment_sofa(self, image: np.ndarray) -> Tuple[np.ndarray, BoundingBox]:
        """
        Segment sofa from image using GrabCut algorithm.

        Args:
            image: Input BGR image

        Returns:
            Tuple[np.ndarray, BoundingBox]: (Segmented image, bounding box)

        Raises:
            Exception: If segmentation fails
            ValueError: If input image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid")

        try:
            height, width = image.shape[:2]
            mask = self._create_initial_mask(height, width)

            rect = (0, 0, width, height)
            # Initialize models
            background_model = np.zeros((1, 65), np.float64)
            foreground_model = np.zeros((1, 65), np.float64)

            # Perform GrabCut segmentation using mask initialization
            cv2.grabCut(
                image,
                mask,
                rect,
                background_model,
                foreground_model,
                self.iterations,
                cv2.GC_INIT_WITH_MASK,
            )

            # Create binary mask and apply it
            binary_mask = np.where(
                (mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1
            ).astype("uint8")
            segmented_image = cv2.bitwise_and(image, image, mask=binary_mask)

            bbox = self._get_bounding_box(binary_mask)
            return segmented_image, BoundingBox(*bbox)

        except cv2.error as e:
            self.logger.error(f"GrabCut segmentation failed: {e}")
            raise Exception(f"Sofa segmentation failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error during segmentation: {e}")
            raise Exception(f"Unexpected error during segmentation: {e}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess sofa image using GrabCut-based segmentation.

        Args:
            image: Input BGR image

        Returns:
            np.ndarray: Processed image with background removed and cropped

        Raises:
            Exception: If critical processing error occurs
        """
        try:
            if image is None or image.size == 0:
                raise ValueError("Input image is empty or invalid")

            # Process image at reduced size for efficiency
            self.logger.debug("Resizing image for processing")
            resized_image, scale = self._resize_image(image)

            self.logger.debug("Performing sofa segmentation")
            segmented_image, bbox = self._segment_sofa(resized_image)

            # Scale coordinates back to original size if needed
            if scale != 1.0:
                self.logger.debug("Scaling results back to original size")
                bbox = BoundingBox(
                    *[
                        int(val / scale)
                        for val in (bbox.x, bbox.y, bbox.width, bbox.height)
                    ]
                )
                segmented_image = cv2.resize(
                    segmented_image,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            # Add padding and ensure coordinates are within image bounds
            x = max(0, bbox.x - self.padding)
            y = max(0, bbox.y - self.padding)
            w = min(image.shape[1] - x, bbox.width + 2 * self.padding)
            h = min(image.shape[0] - y, bbox.height + 2 * self.padding)

            # Crop to sofa region
            self.logger.debug("Cropping to sofa region")
            result = segmented_image[y : y + h, x : x + w]

            if result.size == 0:
                raise Exception("Resulting image is empty after processing")

            return result

        except (ValueError, Exception) as e:
            self.logger.error(f"Failed to preprocess image: {e}")
            raise
