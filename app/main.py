import sys
import cv2
import numpy as np
import streamlit as st

from config import (
    DATABASE_URL,
    ROOT_DIR,
)

sys.path.append(ROOT_DIR)

from src.db.provider import get_database_provider
from src.db.repos import ProductRepository
from app.services import PreprocessorService, SimilarityService


# Initialize services
@st.cache_resource
def init_services():
    db_provider = get_database_provider(DATABASE_URL)
    product_repo = ProductRepository(db_provider.get_session())
    preprocessor_service = PreprocessorService()
    similarity_service = SimilarityService(product_repo)
    return preprocessor_service, similarity_service

def process_uploaded_image(upload):
    """Convert uploaded image to numpy array.
    
    Args:
        upload: Streamlit UploadedFile object
        
    Returns:
        numpy.ndarray: Image as a numpy array in BGR format
    """
    if upload is None:
        return None
        
    # Read uploaded image
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def find_similar_sofa(image, preprocessor_service, similarity_service):
    try:
        # Preprocess the image
        with st.spinner('Preprocessing image...'):
            processed_img = preprocessor_service.preprocess_image(image)
        
        # Extract features and find similar products
        with st.spinner('Finding similar sofas...'):
            features = similarity_service.extract_features(processed_img)
            similar_product = similarity_service.find_most_similar(features)

        return processed_img, similar_product

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("Sofa Similarity Search")
    st.write("Upload a sofa image to find similar ones in our database!")

    # Initialize services
    preprocessor_service, similarity_service = init_services()

    # File uploader
    uploaded_file = st.file_uploader("Choose a sofa image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, channels="BGR")

        # Process image
        img = process_uploaded_image(uploaded_file)        
        processed_img, similar_product = find_similar_sofa(
            img, preprocessor_service, similarity_service
        )

        if processed_img is None:
            return
            
        if similar_product is None:
            st.error("No similar sofas found in the database.")
            return

        # Display similar product
        product, similarity = similar_product

        if product is None:
            st.error("No similar sofas found in the database.")
            return

        # Display similar product
        st.subheader("Similar Sofa")
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(cv2.imread(str(product.raw_file_path)), channels="BGR")
        with col2:
                st.write(f"Product ID: {product.product_id}")
                st.write(f"Similarity Score: {similarity:.2f}")
                st.write(f"Description: {product.description}")
                st.divider()

if __name__ == "__main__":
    main()