# Sofa Similarity Search

A computer vision application that finds similar sofas based on image comparison. The application uses OpenCV for image processing and feature extraction, and provides a user-friendly interface through Streamlit.

## Project Structure

```
.
├── app/                    # Main application code
│   ├── config/            # Configuration settings
│   ├── services/          # Core services
│   │   ├── preprocessor/  # Image preprocessing service
│   │   └── similarity/    # Similarity search service
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   └── sofas/            # Sofa images
│       ├── test/         # Test images
│       ├── raw/          # Original images
│       ├── processed/    # Preprocessed images
│       └── features/     # Extracted features
├── notebooks/            # Jupyter notebooks
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── db/               # Database layer
│   ├── feature_extractor/# Feature extraction
│   └── preprocessor/     # Image preprocessing
├── tests/                # Test suite
└── manage.py             # Management script
```

## Getting Started

1. First, create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Initialize the database with sample sofa images:
```bash
python manage.py init-db
```

3. Run the application:
```bash
python manage.py run-app
```

## Management Commands

The project includes a `manage.py` script with several commands to help you manage the application:

### 1. Initialize Database
```bash
python manage.py init-db
```
- Initializes the SQLite database
- Processes raw sofa images
- Extracts features
- Must be run before using the application

### 2. Run Tests
```bash
python manage.py test [path]
```
Options:
- `path`: Optional path to specific test file or directory
- `--coverage` or `-c`: Run tests with coverage report
- `--failfast` or `-x`: Stop on first failure

### 3. Run Application
```bash
python manage.py run-app
```
- Starts the Streamlit web application
- Opens in your default web browser
- Allows uploading and comparing sofa images

### 4. View Documentation
```bash
python manage.py docs
```
- Builds and opens the project documentation
- Contains API reference and usage guides
- Automatically rebuilds if documentation changes

## Usage Flow

1. **Initial Setup**:
   - Run `init-db` to set up the database and process sample images
   - This is required before first use

2. **Development**:
   - Run `test` to ensure everything works correctly
   - Use `clean-imports` to maintain code quality

3. **Running the App**:
   - Use `run-app` to start the Streamlit interface
   - Upload sofa images to find similar ones

4. **Documentation**:
   - Use `docs` to view the API reference and guides
   - Helpful for understanding the codebase

## Git Hooks and Commit Conventions

### Setting up Husky

The project uses Husky to enforce commit message conventions and run pre-commit checks. To set it up:

1. Install npm dependencies:
```bash
npm install
```

2. Prepare Husky:
```bash
npm run prepare
```

This will set up Git hooks in your local repository.

### Commit Message Conventions

We follow the Conventional Commits specification. Each commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semi-colons, etc)
- `refactor`: Code changes that neither fix a bug nor add a feature
- `test`: Adding or modifying tests
- `chore`: Changes to build process or auxiliary tools

Examples:
```
feat(similarity): add color histogram feature extraction
fix(preprocessor): handle empty image input
docs: update README with setup instructions
test(preprocessor): add unit tests for image segmentation
```

## Technical Details

- Image Processing: OpenCV
- Feature Extraction: Color histograms
- Database: SQLite
- Web Interface: Streamlit
- Testing: pytest
- Documentation: Sphinx

## Contributing

1. Ensure tests pass: `python manage.py test`
2. Update documentation if needed
3. Run isort and black in the root directory to ensure proper formatting (PEP8)
4. Submit a pull request
