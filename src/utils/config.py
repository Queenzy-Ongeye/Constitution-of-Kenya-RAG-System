import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PDF_PATH = os.path.join(DATA_DIR, "ken127322.pdf")

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# API settings
OPENAI_API_KEY = os.getenv("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")

# Vector database settings
COLLECTION_NAME = "kenyan_constitution" 