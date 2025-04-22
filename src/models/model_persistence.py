import os
import pickle
import numpy as np
from typing import Tuple
from src.utils.config import PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model_and_embeddings(model, embeddings: np.ndarray, texts: list, metadata: dict = None):
    """
    Save the model, embeddings, and associated data.
    
    Args:
        model: The embedding model to save
        embeddings: The generated embeddings
        texts: The original text chunks
        metadata: Optional metadata about the model
    """
    # Save the model
    model_path = os.path.join(MODEL_DIR, "embedding_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save embeddings and texts
    data = {
        'embeddings': embeddings,
        'texts': texts,
        'metadata': metadata or {}
    }
    data_path = os.path.join(MODEL_DIR, "embeddings_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Model and embeddings saved to {MODEL_DIR}")

def load_model_and_embeddings() -> Tuple[object, np.ndarray, list, dict]:
    """
    Load the saved model and embeddings.
    
    Returns:
        Tuple containing (model, embeddings, texts, metadata)
    """
    model_path = os.path.join(MODEL_DIR, "embedding_model.pkl")
    data_path = os.path.join(MODEL_DIR, "embeddings_data.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Model or embeddings data not found. Please run the model first.")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load embeddings and data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    return model, data['embeddings'], data['texts'], data['metadata'] 