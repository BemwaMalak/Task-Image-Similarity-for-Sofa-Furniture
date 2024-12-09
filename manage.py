#!/usr/bin/env python3

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from src.db.models import Product
from src.db.provider import get_database_provider
from src.feature_extractor import ColorHistogramExtractor
from src.preprocessor import SofaSegmenter


def generate_random_description():
    """Generate a random sofa description."""
    materials = ["leather", "fabric", "velvet", "microfiber", "suede"]
    colors = ["brown", "black", "gray", "beige", "navy"]
    styles = ["modern", "classic", "contemporary", "traditional", "minimalist"]
    features = ["comfortable", "elegant", "spacious", "cozy", "luxurious"]

    return f"A {random.choice(features)} {random.choice(styles)} sofa made of {random.choice(materials)} in {random.choice(colors)}"


def init_db(args):
    """Initialize the database with sofa products."""
    # Create necessary directories
    processed_dir = Path("data/sofas/processed")
    features_dir = Path("data/sofas/features")
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # Initialize database and get session
    db = get_database_provider()
    db.initialize_database()
    session = db.get_session()

    # Initialize preprocessor and feature extractor
    preprocessor = SofaSegmenter()
    feature_extractor = ColorHistogramExtractor()

    # Get all files from raw directory
    raw_dir = Path("data/sofas/raw")
    image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png"))

    try:
        for idx, image_path in enumerate(image_files, 1):
            print(f"Processing image {idx}/{len(image_files)}: {image_path.name}")

            # Read the image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Process image
            try:
                processed_image = preprocessor.preprocess(image)

                # Save processed image
                processed_path = processed_dir / image_path.name
                cv2.imwrite(str(processed_path), processed_image)

                # Extract features
                color_features, descriptors = feature_extractor.extract_features(
                    processed_image
                )

                # Save features
                features_path = features_dir / f"{image_path.stem}.npz"
                np.savez_compressed(
                    features_path,
                    keypoints=color_features,
                    descriptors=descriptors,
                )

                # Create product in database
                product = Product(
                    product_id=image_path.stem,
                    name=f"Sofa {idx}",
                    description=generate_random_description(),
                    raw_file_path=str(image_path),
                    processed_file_path=str(processed_path),
                    features_file_path=str(features_path),
                )
                session.add(product)
                session.commit()

            except Exception as e:
                print(f"Error processing {image_path.name}: {str(e)}")
                session.rollback()
                continue

    finally:
        db.close_session()


def run_tests(args):
    """Run the test suite with pytest."""
    test_path = args.path if args.path else "tests"
    cmd = ["pytest", test_path, "-v"]

    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])

    if args.failfast:
        cmd.append("-x")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Task Image Similarity Management Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("path", nargs="?", help="Path to test file or directory")
    test_parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run tests with coverage report"
    )
    test_parser.add_argument(
        "--failfast", "-x", action="store_true", help="Stop on first failure"
    )

    # Init DB command
    init_db_parser = subparsers.add_parser(
        "init-db", help="Initialize database with sofa products"
    )

    args = parser.parse_args()

    if args.command == "test":
        sys.exit(run_tests(args))
    elif args.command == "init-db":
        init_db(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
