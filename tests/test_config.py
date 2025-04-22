import os
import sys

# Get the absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# Add project root and src directory to Python path
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

# Data directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
REPORTS_DIR = os.path.join(CURRENT_DIR, 'reports')

# Create reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# Test configuration
TEST_ARTISTS = [
    'botticelli.json',
    'da_vinci.json',
    'kahlo.json',
    'monet.json',
    'picasso.json'
]

# Model configuration
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
SPACY_MODEL = 'en_core_web_sm'

# Test parameters
MAX_MEMORY_TURNS = 20
SIMILARITY_THRESHOLD = 0.5
MIN_RESPONSE_LENGTH = 10
MAX_RESPONSE_LENGTH = 500 