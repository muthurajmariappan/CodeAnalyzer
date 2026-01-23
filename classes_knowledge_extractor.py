from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from tools import Tools


class ClassesKnowledgeExtractor(KnowledgeExtractor):

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
            ], rag, tools, token_counter, llm_provider, knowledge_graph, "c-")
        self.repo_url = repo_url
        self.llm_provider = llm_provider
        self.rag = rag
        self.token_counter = token_counter
        self.knowledge_graph = knowledge_graph

    def extract(self, files_content: Dict[str, str]) -> Dict:
        if self.rag is not None:
            print("Analyzing repository with LLM using RAG...")
        else:
            print("Analyzing repository with LLM...")

        # Get model token limits from provider
        max_tokens = self.llm_provider.get_max_tokens()
        reserve_tokens = 3000  # Reserve for prompt and response
        available_tokens = max_tokens - reserve_tokens
        result = {}

        # Use RAG to retrieve relevant chunks for analysis (if available)
        retrieved_chunks = []
        if self.rag is not None:
            analysis_queries = [
                "classes and packages"
            ]

            for query in analysis_queries:
                chunks = self.rag.retrieve_relevant_chunks(query, n_results=100000)
                retrieved_chunks.extend(chunks)

        # Deduplicate chunks by file path
        seen_files = set()
        unique_chunks = retrieved_chunks
        # for chunk in retrieved_chunks:
        #     file_path = chunk['metadata'].get('file_path', '')
        #     if file_path not in seen_files or len(unique_chunks) < 20:
        #         unique_chunks.append(chunk)
        #         seen_files.add(file_path)

        print(f"retrieved chunks {len(retrieved_chunks)} from vector store")
        print(f"unique chunks {len(unique_chunks)} after dedup")

        with open("D:\\self\\CodeAnalyzer\\out\\" + "c-rag.txt", "w", encoding="utf-8") as file:
            file.write(str(retrieved_chunks))

        # Prepare content for LLM
        files_text_parts = []
        current_tokens = 0

        # Add retrieved chunks
        for chunk in unique_chunks:
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
                        "list of classes that are available.\n")

        prompt = f"""Analyze the following codebase from a GitHub repository {self.repo_url} and extract structured knowledge.
The classes of the repository fetched from a vector store are added below. 
Analyze the content and provide concise description of the project.
{files_text}

Please provide a comprehensive analysis in JSON format with the following structure:
{{
    "classList": <list of class names in the repository as an array>,
    "packageList": <list of package names in the repository as an array>,
}}
"""
        result_text = self.execute_prompt(prompt, max_tokens)

        return result