import pytest
import numpy as np
import cv2
from src.preprocessor.main import SofaSegmenter, BoundingBox
from src.preprocessor.interface import SofaSegmentationError

@pytest.fixture
def sofa_segmenter():
    return SofaSegmenter(padding=10, max_size=800, iterations=1)

@pytest.fixture
def sample_image():
    test_image_path = "data/sofas/test/image_1.jpg"
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
    return image

def test_init_valid_parameters():
    segmenter = SofaSegmenter(padding=10, max_size=800, iterations=1)
    assert segmenter.padding == 10
    assert segmenter.max_size == 800
    assert segmenter.iterations == 1

def test_init_invalid_parameters():
    with pytest.raises(ValueError):
        SofaSegmenter(padding=-1)
    with pytest.raises(ValueError):
        SofaSegmenter(max_size=0)
    with pytest.raises(ValueError):
        SofaSegmenter(iterations=0)

def test_resize_image_no_resize_needed(sofa_segmenter):
    sample_image = np.zeros((800, 800, 3), dtype=np.uint8)
    resized, scale = sofa_segmenter._resize_image(sample_image)
    assert scale == 1.0
    assert resized.shape == sample_image.shape

def test_resize_image_with_resize(sofa_segmenter):
    large_image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    resized, scale = sofa_segmenter._resize_image(large_image)
    assert resized.shape[0] == 800 or resized.shape[1] == 800
    assert scale < 1.0

def test_resize_image_invalid_input(sofa_segmenter):
    with pytest.raises(ValueError):
        sofa_segmenter._resize_image(None)
    with pytest.raises(ValueError):
        sofa_segmenter._resize_image(np.array([]))

def test_create_initial_mask(sofa_segmenter):
    mask = sofa_segmenter._create_initial_mask(300, 400)
    assert mask.shape == (300, 400)
    assert np.all(np.isin(mask, [cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD]))

def test_get_bounding_box_empty_mask(sofa_segmenter):
    mask = np.zeros((300, 400), dtype=np.uint8)
    bbox = sofa_segmenter._get_bounding_box(mask)
    assert bbox == (0, 0, 400, 300)

def test_segment_sofa(sofa_segmenter, sample_image):
    segmented, bbox = sofa_segmenter._segment_sofa(sample_image)
    assert isinstance(segmented, np.ndarray)
    assert isinstance(bbox, BoundingBox)
    assert segmented.shape == sample_image.shape
    assert bbox.width > 0 and bbox.height > 0

def test_segment_sofa_invalid_input(sofa_segmenter):
    with pytest.raises(ValueError):
        sofa_segmenter._segment_sofa(None)
    with pytest.raises(ValueError):
        sofa_segmenter._segment_sofa(np.array([]))

def test_preprocess_valid_image(sofa_segmenter, sample_image):
    result = sofa_segmenter.preprocess(sample_image)
    assert isinstance(result, np.ndarray)
    assert result.size > 0
    assert len(result.shape) == 3

def test_preprocess_invalid_input(sofa_segmenter):
    with pytest.raises(ValueError):
        sofa_segmenter.preprocess(None)
    with pytest.raises(ValueError):
        sofa_segmenter.preprocess(np.array([]))

def test_preprocess_corrupted_image(sofa_segmenter):
    # Create an image with invalid dimensions (2D instead of 3D)
    corrupted_image = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(SofaSegmentationError):
        sofa_segmenter.preprocess(corrupted_image)