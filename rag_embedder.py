"""
RAG Embedder Module using LangChain

Handles RAG (Retrieval-Augmented Generation) with embeddings using LangChain.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

# try:
#     # from langchain_openai import OpenAIEmbeddings
#     # from langchain_community.vectorstores import Chroma
#     # from langchain.text_splitter import RecursiveCharacterTextSplitter
#     # from langchain_core.documents import Document
#     LANGCHAIN_AVAILABLE = True
# except ImportError:
#     LANGCHAIN_AVAILABLE = False

# ChromaDB is optional - only needed for RAG
try:
    # from langchain_chroma import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class RAGEmbedder:
    """Handles RAG (Retrieval-Augmented Generation) with embeddings using LangChain."""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize RAG embedder.
        
        Args:
            api_key: OpenAI API key (for embeddings)
            embedding_model: Embedding model to use
        """
        # if not LANGCHAIN_AVAILABLE:
        #     raise RuntimeError(
        #         "LangChain is not installed. RAG functionality requires langchain. "
        #         "Install it with: pip install langchain langchain-openai langchain-community langchain-chroma"
        #     )
        
        if not CHROMADB_AVAILABLE:
            raise RuntimeError(
                "ChromaDB is not installed. RAG functionality requires chromadb. "
                "Install it with: pip install chromadb, or use --no-rag flag."
            )
        
        # Initialize embeddings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided for embeddings. Set OPENAI_API_KEY environment variable."
            )
        
        self.embedding_model = embedding_model
        # self.embeddings = OpenAIEmbeddings(
        #     model=embedding_model,
        #     openai_api_key=self.api_key
        # )
        self.embeddings = OllamaEmbeddings(
            model="embeddinggemma",#llama3.2:1b
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store
        self.vectorstore = None
        self.temp_db_path = None
    
    def initialize_db(self, repo_name: str):
        """Initialize ChromaDB vector store for this repository."""
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("ChromaDB is not available. Cannot initialize database.")
        
        # self.temp_db_path = tempfile.mkdtemp(prefix="chroma_db_")
        self.temp_db_path = "D:\\self\\CodeAnalyzer\\chroma"
        
        # Create collection name
        collection_name = f"repo_{repo_name.replace('/', '_').replace('-', '_')}"
        
        # Initialize empty vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.temp_db_path
        )
    
    def embed_and_store(self, file_path: str, content: str):
        """
        Embed file content and store in vector database.
        
        Args:
            file_path: Path to the file
            content: File content
        """
        if not self.vectorstore:
            raise RuntimeError("Database not initialized. Call initialize_db() first.")
        
        # Create documents with metadata
        # Use split_documents or create_documents for proper Document objects
        print(f"inside embed_and_store for {file_path}")
        documents = self.text_splitter.create_documents(
            texts=[content],
            metadatas=[{
                "file_path": file_path,
                "source": file_path
            }]
        )
        
        # Add documents to vector store
        if documents:
            print(f"adding documents to vector store inside embed_and_store for {file_path}")
            self.vectorstore.add_documents(documents)
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using semantic search.
        
        Args:
            query: Query string
            n_results: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        if not self.vectorstore:
            return []
        
        try:
            # Use similarity search with metadata
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=n_results
            )
            
            # Format results
            retrieved_chunks = []
            for doc, score in results:
                retrieved_chunks.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance": float(score)  # Lower is better
                })
            
            return retrieved_chunks
        except Exception as e:
            print(f"Warning: Failed to retrieve chunks: {e}")
            return []
    
    def cleanup(self):
        """Clean up temporary database."""
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            shutil.rmtree(self.temp_db_path, ignore_errors=True)
        self.vectorstore = None
