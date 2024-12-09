import sys
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import streamlit as st
from config import DATABASE_URL, ROOT_DIR

sys.path.append(ROOT_DIR)

from services import (
    IPreprocessorService,
    ISimilarityService,
    PreprocessorService,
    SimilarityService,
)
from src.db.models import Product
from src.db.provider import get_database_provider
from src.db.repos import ProductRepository
from utils.image import process_uploaded_image


class SofaSimilarityApp:
    """Main application class for Sofa Similarity Search."""

    def __init__(
        self,
        preprocessor_service: IPreprocessorService,
        similarity_service: ISimilarityService,
    ):
        """
        Initialize the application with required services.

        Args:
            preprocessor_service: Service for preprocessing images
            similarity_service: Service for finding similar sofas
        """
        self.preprocessor_service = preprocessor_service
        self.similarity_service = similarity_service

    def find_similar_sofa(
        self, image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[Product, float]]]:
        """
        Process an image and find similar sofas.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (processed_image, (similar_product, similarity_score))
        """
        try:
            with st.spinner("Preprocessing image..."):
                processed_img = self.preprocessor_service.preprocess_image(image)

            with st.spinner("Finding similar sofas..."):
                features = self.similarity_service.extract_features(processed_img)
                similar_product = self.similarity_service.find_most_similar(features)

            return processed_img, similar_product

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None, None

    def display_similar_product(
        self, product: Product, similarity: float, col1: Any, col2: Any
    ) -> None:
        """
        Display the similar product information in the UI.

        Args:
            product: Product to display
            similarity: Similarity score
            col1: First column for image
            col2: Second column for details
        """
        with col1:
            st.image(cv2.imread(str(product.raw_file_path)), channels="BGR")
        with col2:
            st.write(f"Product ID: {product.product_id}")
            st.write(f"Similarity Score: {similarity:.2f}")
            st.write(f"Description: {product.description}")
            st.divider()

    def run(self) -> None:
        """Run the Streamlit application."""
        st.title("Sofa Similarity Search")
        st.write("Upload a sofa image to find similar ones in our database!")

        uploaded_file = st.file_uploader(
            "Choose a sofa image...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is None:
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, channels="BGR")

        img = process_uploaded_image(uploaded_file)

        if img is None:
            st.error("Failed to process uploaded image.")
            return
            
        processed_img, similar_product = self.find_similar_sofa(img)

        if processed_img is None or similar_product is None:
            st.error("No similar sofas found in the database.")
            return

        product, similarity = similar_product
        if product is None:
            st.error("No similar sofas found in the database.")
            return

        st.subheader("Similar Sofa")
        col1, col2 = st.columns([1, 3])
        self.display_similar_product(product, similarity, col1, col2)


@st.cache_resource
def init_services() -> Tuple[IPreprocessorService, ISimilarityService]:
    """
    Initialize application services.

    Returns:
        Tuple of (preprocessor_service, similarity_service)
    """
    db_provider = get_database_provider(DATABASE_URL)
    product_repo = ProductRepository(db_provider.get_session())
    preprocessor_service = PreprocessorService()
    similarity_service = SimilarityService(product_repo)
    return preprocessor_service, similarity_service


def main() -> None:
    """Main entry point of the application."""
    preprocessor_service, similarity_service = init_services()
    app = SofaSimilarityApp(preprocessor_service, similarity_service)
    app.run()


if __name__ == "__main__":
    main()
