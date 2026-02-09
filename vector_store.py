"""
Vector Store Module
Handles embedding generation and vector similarity search using ChromaDB
"""
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from typing import List, Dict
import os


class VectorStore:
    """Vector database for Tamil document chunks"""
    
    def __init__(self, 
                 collection_name: str = "tamil_docs",
                 persist_directory: str = "./chroma_db",
                 api_key: str = None):
        """
        Initialize vector store with Gemini embedding API
        
        Args:
            collection_name: Name of the collection
            persist_directory: Where to store the database
            api_key: Google API key for Gemini embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize Gemini API for embeddings
        if api_key:
            genai.configure(api_key=api_key)
            print("Using Gemini API for embeddings")
        else:
            raise ValueError("API key required for Gemini embeddings")
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Tamil PDF document chunks"}
            )
            print(f"Created new collection: {collection_name}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using Gemini API"""
        embeddings = []
        print(f"Generating embeddings for {len(texts)} chunks...")
        
        for i, text in enumerate(texts):
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(texts)} chunks")
            except Exception as e:
                print(f"Error embedding chunk {i}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for query using Gemini API"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error embedding query: {e}")
            return [0.0] * 768
    
    def add_documents(self, chunks: List[str], metadata: List[Dict] = None, doc_id: str = None):
        """
        Add document chunks to vector store
        
        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
            doc_id: Document identifier
        """
        if not chunks:
            return
        
        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embed_texts(chunks)
        
        # Generate IDs
        ids = [f"{doc_id}_{i}" if doc_id else f"chunk_{i}" 
               for i in range(len(chunks))]
        
        # Generate metadata
        if metadata is None:
            metadata = [{"chunk_id": i, "doc_id": doc_id} for i in range(len(chunks))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def similarity_search(self, query: str, k: int = 5) -> Dict:
        """
        Search for most similar chunks to query
        
        Args:
            query: User question in Tamil or English
            k: Number of results to return
            
        Returns:
            Dictionary with documents, distances, and metadata
        """
        query_embedding = self.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'distances': results['distances'][0] if results['distances'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'ids': results['ids'][0] if results['ids'] else []
        }
    
    def clear_collection(self):
        """Clear all documents from collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Tamil PDF document chunks"}
        )
        print(f"Cleared collection: {self.collection_name}")
    
    def get_collection_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    def delete_document(self, doc_id: str):
        """Delete all chunks from a specific document"""
        all_items = self.collection.get()
        ids_to_delete = [id for id in all_items['ids'] if id.startswith(doc_id)]
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks from document {doc_id}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Testing Vector Store...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        vs = VectorStore(api_key=api_key)
        print(f"\nVector store initialized successfully")
        print(f"Collection count: {vs.get_collection_count()}")
    else:
        print("GOOGLE_API_KEY not found in .env file")
