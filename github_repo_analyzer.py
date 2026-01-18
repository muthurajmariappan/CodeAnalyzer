#!/usr/bin/env python3
"""
GitHub Repository Knowledge Extractor with RAG and Knowledge Graph

This program accepts a GitHub repository URL, reads its contents,
uses RAG (Retrieval-Augmented Generation) to embed the contents,
stores relationships in a knowledge graph, and uses an LLM to analyze
the repository with token count optimization.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
import subprocess

# OpenAI is optional - only needed if using OpenAI provider
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import modules
from token_counter import TokenCounter
from rag_embedder import RAGEmbedder
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider, create_llm_provider

# Check for LangChain availability
try:
    from langchain_community.vectorstores import Chroma
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class GitHubRepoAnalyzer:
    """Analyzes GitHub repositories using RAG, Knowledge Graph, and LLM."""
    
    def __init__(self, provider_type: str = "openai", model: str = "gpt-4o-mini", 
                 embedding_model: str = "text-embedding-3-small",
                 api_key: Optional[str] = None, ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize the analyzer.
        
        Args:
            provider_type: LLM provider type ('openai' or 'ollama')
            model: LLM model to use (default: gpt-4o-mini for OpenAI, llama3.2 for Ollama)
            embedding_model: Embedding model for RAG (default: text-embedding-3-small, OpenAI only)
            api_key: API key for OpenAI (if using OpenAI provider)
            ollama_base_url: Base URL for Ollama API (if using Ollama provider)
        """
        self.provider_type = provider_type.lower()
        self.model = model
        self.embedding_model = embedding_model
        self.temp_dir = None
        self.rag = None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        # if self.rag is None:
        #     self.rag = RAGEmbedder(self.api_key, self.embedding_model)
        #     self.rag.initialize_db(repo_name)
        
        # Create LLM provider with model
        if self.provider_type == "openai":
            self.llm_provider = create_llm_provider("openai", model=model, rag=self.rag, api_key=api_key)
            # For OpenAI, we still need API key for embeddings
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        elif self.provider_type == "ollama":
            self.llm_provider = create_llm_provider("ollama", model=model, base_url=ollama_base_url)
            # For Ollama, embeddings still use OpenAI (can be extended later)
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError(f"Unknown provider type: {provider_type}. Supported: 'openai', 'ollama'")
        
        # Token counter (works with any provider, but optimized for OpenAI models)
        self.token_counter = TokenCounter(model)
        # self.rag = None  # Will be initialized only if RAG is used
        self.knowledge_graph = KnowledgeGraph()
    
    def clone_repository(self, repo_url: str) -> Path:
        """
        Clone a GitHub repository to a temporary directory.
        
        Args:
            repo_url: GitHub repository URL
            
        Returns:
            Path to the cloned repository
        """
        # Parse the repository URL
        parsed = urlparse(repo_url)
        if not parsed.netloc or "github.com" not in parsed.netloc:
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
        
        # Extract repo name for RAG initialization
        repo_name = parsed.path.strip('/').replace('.git', '')
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="github_repo_")
        repo_path = Path(self.temp_dir)
        
        print(f"Cloning repository to {repo_path}...")
        try:
            # Clone the repository
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print("Repository cloned successfully.")
            
            return repo_path
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Git is not installed. Please install Git to clone repositories."
            )
    
    def read_repository_files(self, repo_path: Path, max_files: int = 100) -> Dict[str, str]:
        """
        Read files from the repository, excluding common ignore patterns.
        
        Args:
            repo_path: Path to the repository
            max_files: Maximum number of files to read
            
        Returns:
            Dictionary mapping file paths to their contents
        """
        files_content = {}
        file_count = 0
        
        # Common ignore patterns
        ignore_patterns = {
            ".git", ".github", "__pycache__", "node_modules", ".venv", "venv",
            "env", ".env", "dist", "build", ".pytest_cache", ".mypy_cache",
            ".idea", ".vscode", "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
            "*.egg-info", ".coverage", "htmlcov", ".tox", ".chroma"
        }
        
        # Common file extensions to include
        code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
            ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
            ".scala", ".clj", ".sh", ".yaml", ".yml", ".json", ".xml",
            ".html", ".css", ".scss", ".md", ".txt", ".dockerfile", ".sql"
        }
        
        print(f"Reading files from repository (max {max_files} files)...")
        
        for root, dirs, filenames in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(
                pattern in d.lower() for pattern in ignore_patterns
            )]
            
            for filename in filenames:
                if file_count >= max_files:
                    print(f"Reached maximum file limit ({max_files}). Stopping.")
                    break
                
                file_path = Path(root) / filename
                relative_path = file_path.relative_to(repo_path)
                
                # Skip ignored files
                if any(pattern in str(relative_path).lower() for pattern in ignore_patterns):
                    continue
                
                # Include code files and important config files
                if file_path.suffix.lower() in code_extensions or filename.lower() in {
                    "dockerfile", "makefile", ".gitignore", ".dockerignore",
                    "requirements.txt", "package.json", "pom.xml", "build.gradle"
                }:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            # Limit file size to avoid token limits
                            if len(content) > 100000:  # ~100KB
                                content = content[:100000] + "\n... (truncated)"
                            files_content[str(relative_path)] = content
                            file_count += 1
                    except Exception as e:
                        print(f"Warning: Could not read {relative_path}: {e}")
        
        print(f"Read {len(files_content)} files from repository.")
        return files_content
    
    def embed_repository(self, files_content: Dict[str, str], repo_name: str):
        """
        Embed all repository files using RAG.
        
        Args:
            files_content: Dictionary mapping file paths to contents
            repo_name: Repository name for database initialization
        """
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError(
                "LangChain is not installed. Cannot use RAG. "
                "Install with: pip install langchain langchain-openai langchain-community langchain-chroma, "
                "or use --no-rag flag."
            )
        
        if self.rag is None:
            self.rag = RAGEmbedder(self.api_key, self.embedding_model)
            self.rag.initialize_db(repo_name)
        
        print("Embedding repository files...")
        total_files = len(files_content)
        
        for i, (file_path, content) in enumerate(files_content.items(), 1):
            # if i % 10 == 0:
            print(f"Embedding progress: {i}/{total_files} files...")
            try:
                if not file_path.endswith("sql"):
                    print(f"embedding {file_path}")
                    # self.rag.embed_and_store(file_path, content)
            except Exception as e:
                print(f"Warning: Failed to embed {file_path}: {e}")
        
        print("Repository embedding complete!")
    
    def extract_knowledge(self, files_content: Dict[str, str], repo_url: str) -> Dict:
        """
        Use LLM with RAG to extract knowledge from repository files.
        
        Args:
            files_content: Dictionary mapping file paths to contents
            repo_url: Original repository URL
            
        Returns:
            Dictionary containing extracted knowledge
        """
        if self.rag is not None:
            print("Analyzing repository with LLM using RAG...")
        else:
            print("Analyzing repository with LLM...")
        
        # Get model token limits from provider
        max_tokens = self.llm_provider.get_max_tokens(self.model)
        reserve_tokens = 3000  # Reserve for prompt and response
        available_tokens = max_tokens - reserve_tokens
        
        # Use RAG to retrieve relevant chunks for analysis (if available)
        retrieved_chunks = []
        if self.rag is not None:
            analysis_queries = [
                "classes and object-oriented design patterns",
                "functions and methods",
                "imports and dependencies",
                "architecture and design patterns",
                "frameworks and libraries used"
            ]
            
            for query in analysis_queries:
                chunks = self.rag.retrieve_relevant_chunks(query, n_results=3)
                retrieved_chunks.extend(chunks)
        
        # Deduplicate chunks by file path
        seen_files = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            file_path = chunk['metadata'].get('file_path', '')
            if file_path not in seen_files or len(unique_chunks) < 20:
                unique_chunks.append(chunk)
                seen_files.add(file_path)
        
        # Also include some key files directly (README, main files, etc.)
        key_files = []
        for path, content in files_content.items():
            path_lower = path.lower()
            if any(keyword in path_lower for keyword in ['readme', 'main', 'app', 'index', 'setup', 'requirements']):
                key_files.append((path, content))
                if len(key_files) >= 5:
                    break
        
        print(f"retrieved chunks {len(retrieved_chunks)} from vector store")
        print(f"unique chunks {len(unique_chunks)} after dedup")
        print(f"key_files {len(key_files)}")

        # Prepare content for LLM
        files_text_parts = []
        current_tokens = 0
        
        # Add key files first
        for path, content in key_files:
            file_text = f"=== File: {path} ===\n{content}\n\n"
            file_tokens = self.token_counter.count_tokens(file_text)
            if current_tokens + file_tokens > available_tokens * 0.4:  # Use 40% for key files
                break
            files_text_parts.append(file_text)
            current_tokens += file_tokens
        
        # Add retrieved chunks
        for chunk in unique_chunks[:15]:  # Limit to 15 chunks
            chunk_text = f"=== File: {chunk['metadata'].get('file_path', 'unknown')} (Relevant Chunk) ===\n{chunk['content']}\n\n"
            chunk_tokens = self.token_counter.count_tokens(chunk_text)
            if current_tokens + chunk_tokens > available_tokens:
                break
            files_text_parts.append(chunk_text)
            current_tokens += chunk_tokens
        
        files_text = "".join(files_text_parts)
        print(f"files content before optimizing {len(files_text)}")
        
        # Optimize if needed
        files_text = self.token_counter.optimize_content(files_text, max_tokens, reserve_tokens)
        print(f"files content after optimizing {len(files_text)}")
        
        # Create a comprehensive prompt
        rag_note = ""
        if self.rag is not None:
            rag_note = "\nThe repository has been analyzed using RAG (Retrieval-Augmented Generation) to identify the most relevant code sections.\n"
        
        prompt = f"""Analyze the following codebase from a GitHub repository ({repo_url}) and extract structured knowledge.{rag_note}
Repository Files and Relevant Code Sections:
{files_text}

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "programming_languages": ["list", "of", "languages", "detected"],
    "frameworks": ["list", "of", "frameworks", "and", "libraries"],
    "classes": [
        {{
            "name": "ClassName",
            "file": "path/to/file.py",
            "description": "Brief description",
            "methods": ["method1", "method2"]
        }}
    ],
    "functions": [
        {{
            "name": "function_name",
            "file": "path/to/file.py",
            "description": "Brief description",
            "parameters": ["param1", "param2"]
        }}
    ],
    "high_level_design": {{
        "architecture": "Description of overall architecture",
        "patterns": ["design patterns used"],
        "key_components": ["component1", "component2"],
        "data_flow": "Description of how data flows through the system"
    }},
    "dependencies": ["list", "of", "key", "dependencies"],
    "summary": "High-level summary of what this repository does"
}}

Be thorough and extract as much information as possible. Focus on:
- All programming languages used
- Major frameworks, libraries, and tools
- Important classes and their purposes
- Key functions and their roles
- Overall architecture and design patterns
- How components interact
"""

        prompt = f"""Analyze the following codebase from a GitHub repository ({repo_url}) and extract structured knowledge.{rag_note}
        Repository Files and Relevant Code Sections:
        {files_text}

        Provide concise description of the codebase in the Github repository based on the given information.
        """

        try:
            # Count tokens in prompt
            prompt_tokens = self.token_counter.count_tokens(prompt)
            print(f"Prompt tokens: {prompt_tokens}, Max tokens: {max_tokens}")
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert software engineer and code analyst. You analyze codebases and extract structured knowledge about their architecture, design, and implementation. Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Calculate max response tokens
            max_response_tokens = min(4000, max_tokens - prompt_tokens - 100)
            
            # Generate response using LangChain provider
            # For JSON format, add instruction to prompt if not OpenAI
            if self.provider_type != "openai":
                messages[-1]["content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any text outside the JSON object."
            
            print("invoking llm")
            result_text = self.llm_provider.invoke_with_messages(
                messages=messages,
                temperature=0.3,
                max_tokens=max_response_tokens
            )

            print(f"response from llm {str(result_text)}")
            print(f"response from llm {result_text.content}")
            
            # Parse JSON response
            knowledge = json.loads(result_text)
            
            # Build knowledge graph from extracted knowledge
            self.knowledge_graph.extract_from_knowledge(knowledge, repo_url)
            
            # Add knowledge graph to results
            knowledge["knowledge_graph"] = self.knowledge_graph.to_dict()
            
            print("Analysis complete!")
            return knowledge
            
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Response was: {result_text[:500]}")
            raise RuntimeError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract knowledge with LLM: {e}")
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        if self.rag is not None:
            self.rag.cleanup()
    
    def analyze(self, repo_url: str, max_files: int = 100, use_rag: bool = True) -> Dict:
        """
        Main method to analyze a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL
            max_files: Maximum number of files to analyze
            use_rag: Whether to use RAG for embedding and retrieval
            
        Returns:
            Dictionary containing extracted knowledge and knowledge graph
        """
        try:
            # Clone repository
            repo_path = self.clone_repository(repo_url)
            
            # Read files
            files_content = self.read_repository_files(repo_path, max_files)
            
            if not files_content:
                raise ValueError("No files found in repository.")
            
            # Extract repo name for RAG initialization
            parsed = urlparse(repo_url)
            repo_name = parsed.path.strip('/').replace('.git', '')
            
            # Embed repository using RAG
            if use_rag:
                self.embed_repository(files_content, repo_name)
            
            # Extract knowledge using LLM with RAG
            knowledge = self.extract_knowledge(files_content, repo_url)
            
            return knowledge
            
        finally:
            # Always cleanup
            self.cleanup()


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze a GitHub repository using RAG, Knowledge Graph, and LLM"
    )
    parser.add_argument(
        "repo_url",
        help="GitHub repository URL (e.g., https://github.com/user/repo)"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="ollama",
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--model",
        help="LLM model to use (default: gpt-4o-mini for OpenAI, llama3.2 for Ollama)",
        default=None
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
        default=None
    )
    parser.add_argument(
        "--ollama-url",
        help="Ollama API base URL (default: http://localhost:11434)",
        default="http://localhost:11434"
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model for RAG (default: text-embedding-3-small, OpenAI only)",
        default="text-embedding-3-small"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help="Maximum number of files to analyze (default: 100)"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (embedding and retrieval)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path (default: print to stdout)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Set default model based on provider
    if args.model is None:
        if args.provider == "openai":
            args.model = "gpt-4o-mini"
        elif args.provider == "ollama":
            args.model = "llama3.2"
    
    try:
        # Create analyzer
        analyzer = GitHubRepoAnalyzer(
            provider_type=args.provider,
            model=args.model,
            embedding_model=args.embedding_model,
            api_key=args.api_key,
            ollama_base_url=args.ollama_url
        )
        
        # Analyze repository
        knowledge = analyzer.analyze(
            args.repo_url,
            max_files=args.max_files,
            use_rag=not args.no_rag
        )
        
        # Output results
        output_json = json.dumps(knowledge, indent=2)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"\nResults saved to {args.output}")
        else:
            print("\n" + "="*80)
            print("EXTRACTED KNOWLEDGE")
            print("="*80)
            print(output_json)
        
        # Print knowledge graph statistics
        if "knowledge_graph" in knowledge:
            stats = knowledge["knowledge_graph"].get("statistics", {})
            print("\n" + "="*80)
            print("KNOWLEDGE GRAPH STATISTICS")
            print("="*80)
            print(f"Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"Total Edges: {stats.get('total_edges', 0)}")
            print(f"Entity Types: {json.dumps(stats.get('entity_types', {}), indent=2)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
