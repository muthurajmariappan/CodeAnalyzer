from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from knowledge_graph import KnowledgeGraph
from tools import Tools
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter


class SampleToolKnowledgeExtractor(KnowledgeExtractor):

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
            ], rag, tools, token_counter, llm_provider, knowledge_graph, "st-")
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

        prompt = f"""Analyze the codebase from the GitHub repository {self.repo_url} and extract structured 
knowledge. The contents of the repository are parsed and stored in a vector store. There are tools available 
to fetch details from the vector store.
For any query, use the tools to fetch relevant information. 

Fetch details about class ActorController and provide a comprehensive analysis in JSON format with the following structure:
{{
    "description" : <a brief description of the class>,
    "package" : <the package to which this class belongs>,
    "methods" : <list of methods in the class as an array>,
    "dependencies" : <list of dependencies that this class has> 
}}
"""
        result_text = self.execute_prompt_with_tools(prompt)

        return result