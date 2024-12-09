import cv2
import numpy as np
import pytest

from src.feature_extractor import ColorHistogramExtractor


@pytest.fixture
def feature_extractor():
    return ColorHistogramExtractor(hist_bins=32)


@pytest.fixture
def sample_image():
    test_image_path = "data/sofas/test/image_1.jpg"
    image = cv2.imread(test_image_path)
    if image is None:
        raise FileNotFoundError(f"Test image not found at {test_image_path}")
    return image


def test_init_valid_parameters():
    extractor = ColorHistogramExtractor(hist_bins=32)
    assert extractor.hist_bins == 32


def test_init_invalid_parameters():
    with pytest.raises(ValueError, match="hist_bins must be a positive integer"):
        ColorHistogramExtractor(hist_bins=0)
    with pytest.raises(ValueError, match="hist_bins must be a positive integer"):
        ColorHistogramExtractor(hist_bins=-1)


def test_extract_features_valid_image(feature_extractor, sample_image):
    keypoints, descriptors = feature_extractor.extract_features(sample_image)

    # Check return types
    assert isinstance(keypoints, np.ndarray)
    assert isinstance(descriptors, np.ndarray)

    # Check histogram dimensions
    expected_hist_size = feature_extractor.hist_bins * 3  # H, S, V channels
    assert keypoints.shape == (expected_hist_size, 1)


def test_extract_features_invalid_input(feature_extractor):
    # Test None input
    with pytest.raises(TypeError, match="Input image must be a numpy array"):
        feature_extractor.extract_features(None)

    # Test empty array
    with pytest.raises(
        ValueError, match="Invalid input image: must be a non-empty 3-channel image"
    ):
        feature_extractor.extract_features(np.array([]))

    # Test 2D array (invalid channels)
    with pytest.raises(
        ValueError, match="Invalid input image: must be a non-empty 3-channel image"
    ):
        feature_extractor.extract_features(np.zeros((100, 100)))


def test_compute_similarity_matrix(feature_extractor, sample_image):
    # Extract features from the same image twice
    features1 = feature_extractor.extract_features(sample_image)
    features2 = feature_extractor.extract_features(sample_image)

    # Create a list of feature tuples
    features_list = [
        (features1[0], features1[1]),  # (color_hist, texture_features)
        (features2[0], features2[1]),  # (color_hist, texture_features)
    ]

    similarity_matrix = feature_extractor.compute_similarity_matrix(features_list)

    # Check matrix properties
    assert isinstance(similarity_matrix, np.ndarray)
    assert similarity_matrix.shape == (2, 2)
    assert np.allclose(similarity_matrix, similarity_matrix.T)  # Symmetric
    assert np.allclose(np.diag(similarity_matrix), 1.0)  # Self-similarity = 1
    assert np.all(similarity_matrix >= 0) and np.all(
        similarity_matrix <= 1
    )  # Range [0,1]


def test_compute_similarity_matrix_invalid_input(feature_extractor):
    with pytest.raises(TypeError, match="features_list must be a list"):
        feature_extractor.compute_similarity_matrix(None)

    with pytest.raises(ValueError, match="features_list cannot be empty"):
        feature_extractor.compute_similarity_matrix([])


def test_compute_pair_similarity(feature_extractor, sample_image):
    # Extract features from the same image
    features1 = feature_extractor.extract_features(sample_image)
    features2 = feature_extractor.extract_features(sample_image)

    # Create feature tuples
    feature_tuple1 = (features1[0], features1[1])  # (color_hist, texture_features)
    feature_tuple2 = (features2[0], features2[1])  # (color_hist, texture_features)

    similarity = feature_extractor._compute_pair_similarity(
        feature_tuple1, feature_tuple2
    )

    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1
    assert np.isclose(similarity, 1.0)  # Same image should have similarity = 1


def test_compute_pair_similarity_invalid_input(feature_extractor):
    with pytest.raises(TypeError, match="Features must be tuples"):
        feature_extractor._compute_pair_similarity(None, None)

    with pytest.raises(ValueError, match="Features must contain exactly 2 elements"):
        feature_extractor._compute_pair_similarity((np.array([]),), (np.array([]),))


def test_find_most_similar(feature_extractor, sample_image):
    # Create a database with three images: original, modified, and very different
    features1 = feature_extractor.extract_features(sample_image)
    feature_tuple1 = (features1[0], features1[1])

    # Slightly modified image
    modified_image = sample_image.copy()
    modified_image[0:25, 0:25] = [128, 128, 128]
    features2 = feature_extractor.extract_features(modified_image)
    feature_tuple2 = (features2[0], features2[1])

    # Very different image
    different_image = np.full_like(sample_image, 255)  # White image
    features3 = feature_extractor.extract_features(different_image)
    feature_tuple3 = (features3[0], features3[1])

    database_features = [feature_tuple1, feature_tuple2, feature_tuple3]

    # Query with the original image
    best_idx, similarity_score = feature_extractor.find_most_similar(
        feature_tuple1, database_features
    )

    assert best_idx == 0  # Should match the original image
    assert similarity_score == 1.0  # Perfect match


def test_find_most_similar_invalid_input(feature_extractor):
    with pytest.raises(TypeError, match="query_features must be a tuple"):
        feature_extractor.find_most_similar(None, [])

    with pytest.raises(TypeError, match="database_features must be a list"):
        feature_extractor.find_most_similar((np.array([]), np.array([])), None)

    with pytest.raises(ValueError, match="database_features cannot be empty"):
        feature_extractor.find_most_similar((np.array([]), np.array([])), [])


def test_glcm_calculation(feature_extractor, sample_image):
    gray_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    glcm = feature_extractor._compute_glcm(gray_image)

    assert isinstance(glcm, np.ndarray)
    assert glcm.shape == (32, 32)
    assert np.isclose(np.sum(glcm), 1.0)  # GLCM should be normalized


def test_glcm_calculation_invalid_input(feature_extractor):
    with pytest.raises(TypeError, match="gray_image must be a numpy array"):
        feature_extractor._compute_glcm(None)

    with pytest.raises(ValueError, match="gray_image must be a 2D array"):
        feature_extractor._compute_glcm(np.zeros((100, 100, 3)))


def test_glcm_features(feature_extractor, sample_image):
    gray_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
    glcm = feature_extractor._compute_glcm(gray_image)
    features = feature_extractor._compute_glcm_features(glcm)

    assert isinstance(features, np.ndarray)
    assert features.shape == (
        5,
    )  # 5 features: contrast, homogeneity, energy, correlation, homogeneity
    assert np.all(features >= 0)  # All features should be non-negative


def test_glcm_features_invalid_input(feature_extractor):
    with pytest.raises(TypeError, match="GLCM must be a numpy array"):
        feature_extractor._compute_glcm_features(None)

    with pytest.raises(ValueError, match="GLCM must be a square 2D array"):
        feature_extractor._compute_glcm_features(np.zeros((8, 7)))
