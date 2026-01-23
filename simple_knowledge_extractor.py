from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from tools import Tools


class SimpleKnowledgeExtractor(KnowledgeExtractor):

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
            ], rag, tools, token_counter, llm_provider, knowledge_graph, "s-")
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

        key_files = []
        for path, content in files_content.items():
            path_lower = path.lower()
            if any(keyword in path_lower for keyword in ['readme']):
                if not path_lower.__contains__("ko-kr"):
                    print(f"{path} added to key files list")
                    key_files.append((path, content))
                if len(key_files) >= 5:
                    break

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

        files_text = "".join(files_text_parts)
        print(f"files content before optimizing {len(files_text)}")

        # Optimize if needed
        files_text = self.token_counter.optimize_content(files_text, max_tokens, reserve_tokens)
        print(f"files content after optimizing {len(files_text)}")

        prompt = f"""Analyze the following codebase from a GitHub repository {self.repo_url} and extract structured knowledge.
The README file contents in markdown format of the repository are added below. 
Based on the information provided, think and provide an appropriate description of this repository.
The description should include the purpose of the repository and high level technical details like architecture, frameworks, programming languages.
{files_text}
"""
        self.execute_prompt(prompt, max_tokens)

        return result
