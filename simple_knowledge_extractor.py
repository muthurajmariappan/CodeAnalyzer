import json
from typing import Dict

from base_knowledge_extractor import KnowledgeExtractor
from llm_providers import LLMProvider
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from knowledge_graph import KnowledgeGraph

class SimpleKnowledgeExtractor(KnowledgeExtractor):

    def __init__(self,
                 repo_url: str,
                 llm_provider: LLMProvider,
                 rag: RAGEmbedder,
                 token_counter: TokenCounter,
                 knowledge_graph: KnowledgeGraph):
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

        prompt = f"""Analyze the following codebase from a GitHub repository (https://github.com/codejsha/spring-rest-sakila) and extract structured knowledge.
The README file contents in markdown format of the repository are added below. 
Analyze the content and provide concise description of the project.
{files_text}
"""
        self.execute_prompt(prompt, max_tokens)

        return result

    def execute_prompt(self, prompt: str, max_tokens: int):
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
            if self.llm_provider.get_provider_type() != "openai":
                messages[-1][
                    "content"] += "\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any text outside the JSON object."

            with open("D:\\self\\CodeAnalyzer\\out\\" + "s-prompt.txt", "w", encoding='utf-8') as file:
                file.write(prompt)

            with open("D:\\self\\CodeAnalyzer\\out\\" + "s-messages.txt", "w", encoding='utf-8') as file:
                file.write(str(messages))

            print("invoking llm")
            result_text = self.llm_provider.invoke_with_messages(
                messages=messages,
                temperature=0.3,
                max_tokens=max_response_tokens
            )

            print(f"response from llm {str(result_text)}")
            with open("D:\\self\\CodeAnalyzer\\out\\" + "s-llm-response.txt", "w", encoding='utf-8') as file:
                file.write(str(result_text))

            # Parse JSON response
            knowledge = json.loads(result_text)

            # Build knowledge graph from extracted knowledge
            self.knowledge_graph.extract_from_knowledge(knowledge, self.repo_url)

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

    def initialize_messages(self) -> []:
        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert software engineer and code analyst. You analyze codebases and extract structured knowledge about their architecture, design, and implementation. Always respond with valid JSON."
            },
            {
                "role": "user",
                "content": ""
            }
        ]
        return messages