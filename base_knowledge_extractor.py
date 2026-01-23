import json
from abc import ABC, abstractmethod
from typing import Dict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from llm_providers import LLMProvider
from knowledge_graph import KnowledgeGraph
from tools import Tools


class KnowledgeExtractor(ABC):

    def __init__(self,
                 messages: [],
                 rag: RAGEmbedder,
                 tools: Tools,
                 token_counter: TokenCounter,
                 llm_provider: LLMProvider,
                 knowledge_graph: KnowledgeGraph,
                 prefix: str):
        self.messages = messages
        self.rag = rag
        self.token_counter = token_counter
        self.llm_provider = llm_provider
        self.knowledge_graph = knowledge_graph
        self.tools = tools
        self.prefix = prefix

    @abstractmethod
    def extract(self, files_content: Dict[str, str]) -> Dict:
        pass

    def execute_prompt(self, prompt: str, max_tokens: int) -> str:
        try:
            # Count tokens in prompt
            prompt_tokens = self.token_counter.count_tokens(prompt)
            print(f"Prompt tokens: {prompt_tokens}, Max tokens: {max_tokens}")

            # Calculate max response tokens
            max_response_tokens = min(4000, max_tokens - prompt_tokens - 100)

            self.messages[1]["content"] += prompt

            # Generate response using LangChain provider
            # For JSON format, add instruction to prompt if not OpenAI
            if self.llm_provider.get_provider_type() != "openai":
                self.messages[-1][
                    "content"] += ("\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any text outside the "
                                   "JSON object.")

            with open("D:\\self\\CodeAnalyzer\\out\\" + self.prefix + "-prompt.txt", "w", encoding='utf-8') as file:
                file.write(prompt)

            with open("D:\\self\\CodeAnalyzer\\out\\" + self.prefix + "-messages.txt", "w", encoding='utf-8') as file:
                file.write(str(self.messages))

            print("invoking llm")
            result_text = self.llm_provider.invoke_with_messages(
                messages=self.messages,
                # temperature=0.3,
                # max_tokens=max_response_tokens
            )

            print(f"response from llm {str(result_text)}")
            with open("D:\\self\\CodeAnalyzer\\out\\" + self.prefix + "-llm-response.txt", "w", encoding='utf-8') as file:
                file.write(str(result_text))

            # Parse JSON response
            # knowledge = json.loads(result_text)
            #
            # # Build knowledge graph from extracted knowledge
            # self.knowledge_graph.extract_from_knowledge(knowledge, self.repo_url)
            #
            # # Add knowledge graph to results
            # knowledge["knowledge_graph"] = self.knowledge_graph.to_dict()

            print("Analysis complete!")

            return result_text

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Response was: {result_text[:500]}")
            raise RuntimeError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to extract knowledge with LLM: {e}")

    def execute_prompt_with_tools(self, prompt: str):
        print(f"execute_prompt_with_tools -> {prompt}")
        messages = [self.messages[0], HumanMessage(prompt)]
        self.execute(messages)
        print(f"final messages list -> {messages}")

    def execute(self, llm_messages):
        ai_response = self.llm_provider.get_llm().invoke(llm_messages)
        print(f"ai_response -> {ai_response}")
        llm_messages.append(ai_response)

        if not ai_response.tool_calls:
            return

        for tool_call in ai_response.tool_calls:
            print(f"executing {tool_call["name"].lower()}")
            selected_tool = self.tools.tools_list().get(tool_call["name"].lower())
            tool_response = selected_tool.invoke(tool_call["args"])
            llm_messages.append(ToolMessage(tool_response, tool_call_id=tool_call["id"]))

        self.execute(llm_messages)

