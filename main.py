# Add src directory to Python path
from pathlib import Path
import sys
import os
from dotenv import load_dotenv # type: ignore
import argparse
from tqdm import tqdm

# Load environment variables
load_dotenv()

src_path = str(Path(__file__).parent / "src")
sys.path.append(src_path)

from api.rag_system import RAGSystem
from models.model_persistence import save_model_and_embeddings, load_model_and_embeddings
from processing.document_processor import DocumentProcessor
from models.embeddings import EmbeddingModel
from utils.config import PDF_PATH, EMBEDDING_MODEL, COLLECTION_NAME

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data",
        "models",
        "src/api",
        "src/models",
        "src/processing",
        "src/utils"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def process_documents():
    """Process the constitution document and generate embeddings."""
    print("\n=== Processing Documents ===")
    processor = DocumentProcessor(PDF_PATH)
    
    print("Extracting text from PDF...")
    text = processor.extract_text()
    print(f"Extracted {len(text)} characters")
    
    print("Chunking text...")
    chunks = list(processor.chunk_text(text))  # Still lists, but generator handles large splits better
    print(f"Generated {len(chunks)} chunks")
    
    return chunks

def generate_embeddings(chunks):
    """Generate embeddings for the document chunks."""
    print("\n=== Generating Embeddings ===")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL)
    
    print("Creating embeddings...")
    embeddings = embedding_model.generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    return embedding_model, embeddings

def save_model(embedding_model, embeddings, chunks):
    """Save the model and embeddings."""
    print("\n=== Saving Model ===")
    metadata = {
        'model_name': EMBEDDING_MODEL,
        'num_chunks': len(chunks),
        'collection_name': COLLECTION_NAME
    }
    save_model_and_embeddings(embedding_model.model, embeddings, chunks, metadata)

def test_system():
    """Test the RAG system with sample queries."""
    print("\n=== Testing System ===")
    rag_system = RAGSystem(load_saved=True)
    
    test_questions = [
        "What are the fundamental rights in the Kenyan Constitution?",
        "What is the structure of the Kenyan government?",
        "What are the duties of the President?",
        "How are judges appointed in Kenya?",
        "What are the principles of land policy in Kenya?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = rag_system.query(question)
        for i, result in enumerate(results[:3], 1):  # Show top 3 results
            print(f"\n{i}. {result['text'][:200]}...")  # Show first 200 chars
            print(f"   Distance: {result['distance']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run the Kenyan Constitution RAG System')
    parser.add_argument('--skip-processing', action='store_true', help='Skip document processing and use saved model')
    args = parser.parse_args()
    
    print("=== Kenyan Constitution RAG System ===")
    setup_directories()
    
    if not args.skip_processing:
        # Process documents and generate embeddings
        chunks = process_documents()
        embedding_model, embeddings = generate_embeddings(chunks)
        save_model(embedding_model, embeddings, chunks)
    
    # Test the system
    test_system()

if __name__ == "__main__":
    main()