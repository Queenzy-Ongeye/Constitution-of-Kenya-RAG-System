import pdfplumber
from typing import List
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        """Extract text from PDF file."""
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def chunk_text(self, text: str, chunk_size=500, overlap=50):
        """Generator to yield text chunks from a large document."""
        text_length = len(text)
        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)
            yield text[start:end]
        start += chunk_size - overlap

    def process_document(self) -> List[str]:
        """Process the entire document and return chunks."""
        text = self.extract_text()
        return self.chunk_text(text) 