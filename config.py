"""
Configuration file for Tamil PDF QA System
Centralizes all configuration parameters
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the Tamil PDF QA System"""
    
    # API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    
    # Gemini Model Configuration
    GEMINI_MODEL = "gemini-1.5-flash"  # Options: gemini-1.5-flash, gemini-1.5-pro
    GEMINI_TEMPERATURE = 0.3  # Lower = more factual, Higher = more creative
    GEMINI_TOP_P = 0.95
    GEMINI_TOP_K = 40
    GEMINI_MAX_OUTPUT_TOKENS = 2048
    
    # PDF Processing Configuration
    CHUNK_SIZE = 400  # Characters per chunk
    CHUNK_OVERLAP = 100  # Overlap between chunks
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = "models/text-embedding-004"  # Gemini embedding model
    # Using Gemini API for embeddings - no local model download needed!
    
    # Vector Store Configuration
    VECTOR_DB_COLLECTION = "tamil_docs"
    VECTOR_DB_PATH = "./chroma_db"
    
    # Retrieval Configuration
    DEFAULT_TOP_K = 5  # Number of chunks to retrieve
    MAX_TOP_K = 10
    MIN_TOP_K = 3
    
    # UI Configuration
    PAGE_TITLE = "Tamil PDF QA System"
    PAGE_ICON = "📚"
    LAYOUT = "wide"
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB = 50
    ALLOWED_EXTENSIONS = ['pdf']
    TEMP_UPLOAD_DIR = "./temp_uploads"
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FILE = "tamil_pdf_qa.log"
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is not set in .env file")
        
        if cls.CHUNK_SIZE < 100:
            errors.append("CHUNK_SIZE too small (minimum 100)")
        
        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        
        return errors
    
    @classmethod
    def print_config(cls):
        """Print current configuration (excluding sensitive data)"""
        print("\n=== Tamil PDF QA System Configuration ===")
        print(f"Gemini Model: {cls.GEMINI_MODEL}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Chunk Size: {cls.CHUNK_SIZE}")
        print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"Default Top-K: {cls.DEFAULT_TOP_K}")
        print(f"Vector DB Path: {cls.VECTOR_DB_PATH}")
        print(f"API Key Set: {'Yes' if cls.GOOGLE_API_KEY else 'No'}")
        print("=========================================\n")


if __name__ == "__main__":
    # Validate and print configuration
    Config.print_config()
    
    errors = Config.validate()
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Configuration is valid!")
