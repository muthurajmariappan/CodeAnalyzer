from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from tools import Tools


class OverallKnowledgeExtractor(KnowledgeExtractor):

    def __init__(self,
                 repo_url: str,
                 llm_provider: LLMProvider,
                 rag: RAGEmbedder,
                 tools: Tools,
                 token_counter: TokenCounter,
                 knowledge_graph: KnowledgeGraph):
        super().__init__(
            [
                {
                    "role": "system",
                    "content": "You are an expert software engineer and code analyst. You analyze codebases and "
                               "extract structured knowledge about their architecture, design, and implementation. "
                               "Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": ""
                }
            ], rag, tools, token_counter, llm_provider, knowledge_graph, "o-")
        self.repo_url = repo_url
        self.llm_provider = llm_provider
        self.rag = rag
        self.token_counter = token_counter
        self.knowledge_graph = knowledge_graph

    def extract(self, files_content: Dict[str, str]) -> Dict:
        """
        Use LLM with RAG to extract knowledge from repository files.

        Args:
            files_content: Dictionary mapping file paths to contents

        Returns:
            Dictionary containing extracted knowledge
        """
        if self.rag is not None:
            print("Analyzing repository with LLM using RAG...")
        else:
            print("Analyzing repository with LLM...")

        result = {}

        # Get model token limits from provider
        max_tokens = self.llm_provider.get_max_tokens()
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
                print(f"{path} added to key files list")
                key_files.append((path, content))
                if len(key_files) >= 5:
                    break

        print(f"retrieved chunks {len(retrieved_chunks)} from vector store")
        print(f"unique chunks {len(unique_chunks)} after dedup")
        print(f"key_files {len(key_files)}")

        with open("D:\\self\\CodeAnalyzer\\out\\" + "o-rag.txt", "w") as file:
            file.write(str(retrieved_chunks))

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
            rag_note = ("\nThe repository has been analyzed using RAG (Retrieval-Augmented Generation) to identify the "
                        "most relevant code sections.\n")

        prompt = \
            f"""Analyze the following codebase from a GitHub repository ({self.repo_url}) and extract structured 
            knowledge.{rag_note}
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

        result_text = self.execute_prompt(prompt, max_tokens)

        return result
