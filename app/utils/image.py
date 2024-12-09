from typing import Optional
import cv2
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile


def process_uploaded_image(upload: Optional[UploadedFile]) -> Optional[np.ndarray]:
    """Convert uploaded image to numpy array.

    Args:
        upload: Streamlit UploadedFile object

    Returns:
        numpy.ndarray: Image as a numpy array in BGR format
    """
    if upload is None:
        return None

    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img
