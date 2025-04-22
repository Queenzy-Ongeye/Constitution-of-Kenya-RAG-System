from src.processing.document_processor import DocumentProcessor
from src.models.embeddings import EmbeddingModel
from src.models.vector_db import VectorDatabase
from src.models.model_persistence import save_model_and_embeddings, load_model_and_embeddings
from src.utils.config import PDF_PATH, EMBEDDING_MODEL, COLLECTION_NAME
from typing import List, Dict
import os

class RAGSystem:
    def __init__(self, load_saved: bool = False):
        self.document_processor = DocumentProcessor(PDF_PATH)
        self.embedding_model = EmbeddingModel(EMBEDDING_MODEL)
        self.vector_db = VectorDatabase(COLLECTION_NAME)
        
        if load_saved:
            self._load_saved_model()
        else:
            self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system by processing documents and creating embeddings."""
        print("Processing documents...")
        chunks = self.document_processor.process_document()
        print(f"Generated {len(chunks)} chunks")
        
        print("Creating embeddings...")
        embeddings = self.embedding_model.generate_embeddings(chunks)
        
        print("Storing in vector database...")
        self.vector_db.add_documents(chunks)
        
        print("Saving model and embeddings...")
        metadata = {
            'model_name': EMBEDDING_MODEL,
            'num_chunks': len(chunks),
            'collection_name': COLLECTION_NAME
        }
        save_model_and_embeddings(self.embedding_model.model, embeddings, chunks, metadata)
        
        print("System initialized successfully!")
    
    def _load_saved_model(self):
        """Load saved model and embeddings."""
        print("Loading saved model and embeddings...")
        model, embeddings, chunks, metadata = load_model_and_embeddings()
        
        print("Storing in vector database...")
        self.vector_db.add_documents(chunks)
        
        print("System loaded successfully!")
    
    def query(self, question: str, n_results: int = 5) -> List[Dict]:
        """Query the RAG system with a question."""
        return self.vector_db.query(question, n_results)
    
    def get_relevant_context(self, question: str, n_results: int = 3) -> str:
        """Get relevant context for a question."""
        results = self.query(question, n_results)
        context = "\n\n".join([result['text'] for result in results])
        return context 