from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

app = FastAPI()

client = chromadb.PersistentClient(path="chroma_store")
collection = client.get_or_create_collection(name="Kenya_constitution")

embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
def query_constitution(req: QueryRequest):
    try:
        results = collection.query(
            query_texts=[req.question],
            n_results=3,
            embedding_function=embedding_fn
        )
        return {
            "question": req.question,
            "results": [
                {
                    "id": id_,
                    "text": doc,
                    "title": meta.get("title"),
                    "distance": dist
                }
                for id_, doc, meta, dist in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))