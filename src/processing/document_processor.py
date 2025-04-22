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
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks of specified size."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + CHUNK_SIZE, text_length)
            # Ensure we don't cut in the middle of a word
            if end < text_length:
                while end > start and text[end] not in (' ', '\n'):
                    end -= 1
            chunks.append(text[start:end].strip())
            start = end - CHUNK_OVERLAP if end > start else end
            
        return chunks
    
    def process_document(self) -> List[str]:
        """Process the entire document and return chunks."""
        text = self.extract_text()
        return self.chunk_text(text) 