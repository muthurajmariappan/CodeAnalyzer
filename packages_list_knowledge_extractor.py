from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from knowledge_graph import KnowledgeGraph
from tools import Tools
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter


class PackagesListKnowledgeExtractor(KnowledgeExtractor):

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
                               f"Analyze the following codebase from a GitHub repository {repo_url} and extract "
                               "structured knowledge."
                               "The contents of the repository are parsed and stored in a vector store. There are "
                               "tools available to fetch details from the vector store."
                               "DO NOT MAKE UP INFORMATION. USE AVAILABLE TOOLS TO FETCH DATA AND MAKE INFERENCES."
                               "Always respond with valid JSON."
                },
                {
                    "role": "user",
                    "content": ""
                }
            ], rag, tools, token_counter, llm_provider, knowledge_graph, "pl-")
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

        prompt = f"""Fetch details about packages in the repository and provide a comprehensive analysis in JSON format with the following structure.
Only the java package names. DO NOT INCLUDE CLASSES. INCLUDE ONLY UNIQUE VALUES.        
{{
    "packageList" : <list of packages in the repository>
}}
"""
        result_text = self.execute_prompt_with_tools(prompt)

        return result