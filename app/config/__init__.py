import os

# Database configuration
DATABASE_URL = "sqlite:///./products.db"

# Paths configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Image processing configuration
MAX_IMAGE_SIZE = 800  # Maximum dimension for uploaded images
PADDING = 10  # Padding around segmented sofa
GRABCUT_ITERATIONS = 1  # Number of GrabCut iterations
HIST_BINS = 32  # Number of bins for color histogram