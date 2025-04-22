import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import numpy as np

class VectorDatabase:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the vector database."""
        if metadatas is None:
            metadatas = [{} for _ in documents]
            
        ids = [str(i) for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """Query the vector database."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return [{
            'text': doc,
            'metadata': meta,
            'distance': dist
        } for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )] 