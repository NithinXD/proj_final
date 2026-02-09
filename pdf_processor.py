"""
PDF Processing Module
Handles Tamil PDF ingestion, text extraction, and intelligent chunking
"""
import pdfplumber
import fitz  # PyMuPDF
import re
from typing import List, Dict
import unicodedata


class PDFProcessor:
    """Process Tamil PDF documents with Unicode normalization"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for Tamil Unicode)"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(
                        layout=True,
                        x_tolerance=3,
                        y_tolerance=3
                    )
                    if page_text:
                        text += page_text + "\n\n"
        except Exception as e:
            print(f"Error with pdfplumber: {e}")
        return text
    
    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (fallback method)"""
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text += page.get_text() + "\n\n"
            doc.close()
        except Exception as e:
            print(f"Error with PyMuPDF: {e}")
        return text
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text with fallback mechanisms and better Tamil handling"""
        # Try pdfplumber first (better for Tamil)
        text = self.extract_text_pdfplumber(pdf_path)
        
        # Fallback to PyMuPDF if pdfplumber fails or returns gibberish
        if not text.strip() or self._is_gibberish(text):
            print("Trying alternative extraction method...")
            text = self.extract_text_pymupdf(pdf_path)
        
        # If still gibberish or empty, the PDF might be scanned/image-based
        if not text.strip() or self._is_gibberish(text):
            print("⚠️ Warning: PDF appears to be scanned or has encoding issues.")
            print("Please use a text-based PDF with proper Tamil Unicode encoding.")
            return text
        
        # Normalize Unicode (important for Tamil)
        text = unicodedata.normalize('NFC', text)
        
        # Clean text
        text = self._clean_text(text)
        
        return text
    
    def _is_gibberish(self, text: str) -> bool:
        """Check if text contains mostly gibberish/unreadable characters"""
        if not text:
            return True
        
        # Count special/invalid characters
        special_chars = sum(1 for c in text if c in '@#$*&')
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return True
        
        # If more than 20% special chars, likely gibberish
        return (special_chars / total_chars) > 0.2
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix common PDF extraction issues
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def semantic_chunk(self, text: str) -> List[str]:
        """
        Semantic chunking with paragraph/sentence boundaries
        Better for Tamil text than fixed-size chunking
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk_size, finalize current chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap
                current_chunk = self._get_overlap(current_chunk) + para + " "
            else:
                current_chunk += para + "\n\n"
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def sentence_split_tamil(self, text: str) -> List[str]:
        """
        Split Tamil text into sentences
        Tamil uses ., ?, !, and Tamil punctuation
        """
        # Tamil sentence endings: ., ?, !, ।
        sentences = re.split(r'[.!?।]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Complete PDF processing pipeline
        Returns extracted text and chunks
        """
        print(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text(pdf_path)
        
        if not text.strip():
            raise ValueError("No text could be extracted from PDF")
        
        # Create chunks
        chunks = self.semantic_chunk(text)
        
        return {
            'full_text': text,
            'chunks': chunks,
            'num_chunks': len(chunks),
            'total_chars': len(text)
        }


if __name__ == "__main__":
    # Test the processor
    processor = PDFProcessor()
    
    # Example test
    print("PDF Processor module loaded successfully")
    print(f"Default chunk size: {processor.chunk_size}")
    print(f"Default overlap: {processor.chunk_overlap}")
