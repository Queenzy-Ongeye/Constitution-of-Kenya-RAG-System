from fastapi import FastAPI, HTTPException
from api.schema import QueryRequest, QueryResponse, QueryResult
from api.rag_system import RAGSystem

app = FastAPI(title="Kenyan Constitution RAG API")

# Load RAG system once
rag_system = RAGSystem(load_saved=True)

@app.get("/")
def read_root():
    return {"message": "Kenyan Constitution RAG API is live!"}

@app.post("/query", response_model=QueryResponse)
def query_constitution(request: QueryRequest):
    try:
        results = rag_system.query(request.question)
        return QueryResponse(
            results=[
                QueryResult(text=chunk["text"], distance=chunk["distance"])
                for chunk in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
