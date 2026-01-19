"""
RAG Embedder Module using LangChain

Handles RAG (Retrieval-Augmented Generation) with embeddings using LangChain.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma

class RAGEmbedder:
    """Handles RAG (Retrieval-Augmented Generation) with embeddings using LangChain."""
    
    def __init__(self, embeddings: Embeddings, db_path: str = None):
        """
        Initialize RAG embedder.
        
        Args:
            embeddings: instance of Embeddings
        """
        self.embeddings = embeddings

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Vector store
        self.vectorstore = None
        if db_path is None:
            self.temp_db_path = tempfile.mkdtemp(prefix="chroma_db_")
        else:
            self.temp_db_path = db_path
    
    def initialize_db(self, repo_name: str):
        """Initialize ChromaDB vector store for this repository."""
        
        # self.temp_db_path = tempfile.mkdtemp(prefix="chroma_db_")
        # self.temp_db_path = "D:\\self\\CodeAnalyzer\\chroma"
        print(f"the vector db is at {self.temp_db_path}")
        
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
