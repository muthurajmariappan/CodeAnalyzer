"""
LLM Providers Module using LangChain

Abstract interface for different LLM providers (OpenAI, Ollama, etc.) using LangChain.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings


class LLMProvider(ABC):
    """Abstract base class for LLM providers using LangChain."""

    @abstractmethod
    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Invoke the LLM with a list of messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """
        Get maximum context tokens for a model.
        
        Args:
            model: Model name
            
        Returns:
            Maximum context tokens
        """
        pass

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """
        Get the LangChain Embeddings instance

        Returns:
            LangChain Embeddings instance
        """

    @abstractmethod
    def get_model(self) -> str:
        """
        Get the model name

        Returns:
            Model name
        """

    @abstractmethod
    def get_provider_type(self) -> str:
        """
        Get the provider type

        Returns:
            provider type
        """


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name to use
        """
        import os
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai package is required for OpenAI provider. "
                "Install it with: pip install langchain-openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model
        self.embedding_model = embedding_model
        self.llm = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            temperature=0.3
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt."""
        return self.llm.invoke(prompt, **kwargs).content

    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Invoke the LLM with messages."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        # Convert messages to LangChain message format
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        response = self.llm.invoke(langchain_messages, **kwargs)
        return response.content

    def get_max_tokens(self) -> int:
        """Get maximum context tokens for OpenAI models."""
        max_tokens_map = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }
        return max_tokens_map.get(self.model, 4096)  # Default conservative limit

    def get_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.api_key
        )

    def get_model(self) -> str:
        return self.model

    def get_provider_type(self) -> str:
        return "openai"


class OllamaProvider(LLMProvider):
    """Ollama LLM provider using LangChain."""

    def __init__(self, model: str = "llama3.2", embedding_model: str = "embeddinggemma",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            model: Model name to use
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        try:
            from langchain_community.llms import Ollama
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-community package is required for Ollama provider. "
                "Install it with: pip install langchain-community"
            )

        print('#####base_url - ' + base_url)
        print('#####model - ' + model)

        self.model = model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.llm = ChatOllama(
            model=self.model,
            base_url=base_url,
            temperature=0.3
        )

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt."""
        print(f"invoking ollama with {len(prompt)} length")
        return self.llm.invoke(prompt, **kwargs)

    def invoke_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Invoke the LLM with messages."""
        # Convert messages to a single prompt for Ollama
        print(f"inside invoke_with_messages of ollama with {len(messages)} messages")
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(("system", content))
            elif role == "user":
                prompt_parts.append(("human", content))
            elif role == "assistant":
                prompt_parts.append(("Assistant", content))

        print(f"the prompt_parts {len(prompt_parts)}")
        # prompt = "\n\n".join(prompt_parts)
        # print(f"the prompt_parts {prompt_parts}")
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the given sentence.",
            ),
            ("human", "I love programming."),
        ]
        return self.llm.invoke(messages, **kwargs)

    def get_max_tokens(self) -> int:
        """
        Get maximum context tokens for Ollama models.
        
        Note: This is a default value. Actual limits depend on the model.
        Most modern open source models support 32k-128k tokens.
        """
        return 32768  # Default for most modern open source models

    def get_embeddings(self) -> Embeddings:
        return OllamaEmbeddings(
            model=self.embedding_model,  # llama3.2:1b
        )

    def get_model(self) -> str:
        return self.model

    def get_provider_type(self) -> str:
        return "ollama"


def create_llm_provider(provider_type: str, model: str, embedding_model: str, **kwargs) -> LLMProvider:
    """
    Factory function to create an LLM provider.
    
    Args:
        provider_type: Type of provider ('openai' or 'ollama')
        model: Model name to use
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider(
            api_key=kwargs.get("api_key"),
            model=model,
            embedding_model=embedding_model
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model=model,
            embedding_model=embedding_model,
            base_url=kwargs.get("base_url", "http://localhost:11434")
        )
    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            "Supported types: 'openai', 'ollama'"
        )
