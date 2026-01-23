#!/usr/bin/env python3
"""
GitHub Repository Knowledge Extractor with RAG and Knowledge Graph

This program accepts a GitHub repository URL, reads its contents,
uses RAG (Retrieval-Augmented Generation) to embed the contents,
stores relationships in a knowledge graph, and uses an LLM to analyze
the repository with token count optimization.
"""

from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from git_rag_loader import GitRAGLoader
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter


class GitHubRepoAnalyzer:
    """Analyzes GitHub repositories using RAG, Knowledge Graph, and LLM."""

    def __init__(self,
                 llm_provider: LLMProvider,
                 rag: RAGEmbedder,
                 token_counter: TokenCounter,
                 knowledge_graph: KnowledgeGraph,
                 git_rag_loader: GitRAGLoader,
                 knowledge_extractors: [KnowledgeExtractor],
                 repo_url: str,
                 repo_name: str,
                 max_files: int):
        """
        Initialize the analyzer.
        
        Args:
            llm_provider: LLM provider ('openai' or 'ollama')
            rag: RAGEmbedder instance based on the LLM provider
        """
        self.temp_dir = None
        self.repo_url = repo_url
        self.repo_name = repo_name
        self.max_files = max_files
        self.rag = rag
        self.llm_provider = llm_provider
        self.token_counter = token_counter
        self.knowledge_graph = knowledge_graph
        self.git_rag_loader = git_rag_loader
        self.knowledge_extractors = knowledge_extractors

    def cleanup(self):
        """Clean up temporary files."""
        if self.git_rag_loader is not None:
            self.git_rag_loader.cleanup()
        if self.rag is not None:
            self.rag.cleanup()

    def analyze(self) -> Dict:
        """
        Main method to analyze a GitHub repository.
        
        Args:
            
        Returns:
            Dictionary containing extracted knowledge and knowledge graph
        """
        try:
            files_content = self.git_rag_loader.load_files_to_rag()
            # Extract knowledge using LLM with RAG
            for ke in self.knowledge_extractors:
                ke.extract(files_content)

            return {}

        finally:
            # Always cleanup
            self.cleanup()
