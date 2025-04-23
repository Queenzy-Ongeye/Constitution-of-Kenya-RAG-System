from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str

class QueryResult(BaseModel):
    text: str
    distance: float

class QueryResponse(BaseModel):
    results: List[QueryResult]