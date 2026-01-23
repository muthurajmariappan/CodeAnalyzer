import os
import sys
from urllib.parse import urlparse

from git_rag_loader import GitRAGLoader
from github_repo_analyzer import GitHubRepoAnalyzer
from knowledge_graph import KnowledgeGraph
from llm_providers import LLMProvider, create_llm_provider, EmbeddingsProvider, create_embeddings_provider
from overall_knowledge_extractor import OverallKnowledgeExtractor
from simple_knowledge_extractor import SimpleKnowledgeExtractor
from dependencies_knowledge_extractor import DependenciesKnowledgeExtractor
from classes_knowledge_extractor import ClassesKnowledgeExtractor
from sample_tool_knowledge_extractor import SampleToolKnowledgeExtractor
from packages_list_knowledge_extractor import PackagesListKnowledgeExtractor
from rag_embedder import RAGEmbedder
from token_counter import TokenCounter
from tools import Tools


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
        default="openai",
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
        default=None
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10000,
        help="Maximum number of files to analyze (default: 100)"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG (embedding and retrieval)"
    )
    parser.add_argument(
        "--rag-db",
        help="RAG DB path (default: temp directory in temp folder)",
        default=None
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
            args.model = "llama3.2:1b"  # gemma3:270m llama3.2:1b gpt-oss qwen2.5-coder devstral-small-2

    if args.embedding_model is None:
        if args.provider == "openai":
            args.embedding_model = "text-embedding-3-small"
        elif args.provider == "ollama":
            args.embedding_model = "embeddinggemma"

    print(f"args - {args}")

    try:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        repo_name = get_repo_name(args.repo_url)
        embedding_provider = get_embedding_provider(args.provider, args.embedding_model, api_key, args.ollama_url)
        rag = get_rag_embedder(repo_name, embedding_provider, args.rag_db)
        tools = Tools(rag)
        llm_provider = get_llm_provider(tools, args.provider, args.model, api_key, args.ollama_url)
        token_counter = TokenCounter(llm_provider.get_model())
        knowledge_graph = KnowledgeGraph()
        git_rag_loader = GitRAGLoader(args.repo_url, repo_name, args.max_files, rag)

        simple_knowledge_extractor = SimpleKnowledgeExtractor(args.repo_url, llm_provider, rag, tools, token_counter,
                                                              knowledge_graph)
        dependencies_knowledge_extractor = DependenciesKnowledgeExtractor(args.repo_url, llm_provider, rag, tools,
                                                                          token_counter, knowledge_graph)
        classes_knowledge_extractor = ClassesKnowledgeExtractor(args.repo_url, llm_provider, rag, tools, token_counter,
                                                                knowledge_graph)
        overall_knowledge_extractor = OverallKnowledgeExtractor(args.repo_url, llm_provider, rag, tools, token_counter,
                                                                knowledge_graph)
        sample_tool_knowledge_extractor = SampleToolKnowledgeExtractor(args.repo_url, llm_provider, rag, tools, token_counter,
                                                                       knowledge_graph)
        package_list_extractor = PackagesListKnowledgeExtractor(args.repo_url, llm_provider, rag, tools, token_counter,
                                                                        knowledge_graph)

        # Create analyzer
        analyzer = GitHubRepoAnalyzer(
            llm_provider,
            rag,
            token_counter,
            knowledge_graph,
            git_rag_loader,
            [
                # simple_knowledge_extractor,
                # dependencies_knowledge_extractor,
                # classes_knowledge_extractor,
                sample_tool_knowledge_extractor,
                package_list_extractor,
                # overall_knowledge_extractor
            ],
            args.repo_url,
            repo_name,
            args.max_files,
        )

        # Analyze repository
        knowledge = analyzer.analyze()

        # Output results
        output_json = json.dumps(knowledge, indent=2)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_json)
            print(f"\nResults saved to {args.output}")
        else:
            print("\n" + "=" * 80)
            print("EXTRACTED KNOWLEDGE")
            print("=" * 80)
            print(output_json)

        # Print knowledge graph statistics
        if "knowledge_graph" in knowledge:
            stats = knowledge["knowledge_graph"].get("statistics", {})
            print("\n" + "=" * 80)
            print("KNOWLEDGE GRAPH STATISTICS")
            print("=" * 80)
            print(f"Total Nodes: {stats.get('total_nodes', 0)}")
            print(f"Total Edges: {stats.get('total_edges', 0)}")
            print(f"Entity Types: {json.dumps(stats.get('entity_types', {}), indent=2)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def get_llm_provider(tools: Tools, provider_type: str, model: str, api_key: str, ollama_base_url: str):
    # Create LLM provider with model
    if provider_type == "openai":
        llm_provider = create_llm_provider(tools, "openai", model=model, api_key=api_key)
        # For OpenAI, we still need API key for embeddings
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    elif provider_type == "ollama":
        llm_provider = create_llm_provider(tools, "ollama", model=model,
                                           base_url=ollama_base_url)
        # For Ollama, embeddings still use OpenAI (can be extended later)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: 'openai', 'ollama'")

    return llm_provider


def get_embedding_provider(provider_type: str, embedding_model: str, api_key: str, ollama_base_url: str):
    # Create LLM provider with model
    if provider_type == "openai":
        llm_provider = create_embeddings_provider("openai", embedding_model=embedding_model, api_key=api_key)
        # For OpenAI, we still need API key for embeddings
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    elif provider_type == "ollama":
        llm_provider = create_embeddings_provider("ollama", embedding_model=embedding_model,
                                                  base_url=ollama_base_url)
        # For Ollama, embeddings still use OpenAI (can be extended later)
        api_key = api_key or os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: 'openai', 'ollama'")

    return llm_provider


def get_repo_name(repo_url: str) -> str:
    # Extract repo name for RAG initialization
    parsed = urlparse(repo_url)
    repo_name = parsed.path.strip('/').replace('.git', '')
    return repo_name


def get_rag_embedder(repo_name: str, llm_provider: EmbeddingsProvider, db_path: str = None) -> RAGEmbedder:
    embedder = RAGEmbedder(llm_provider.get_embeddings(), db_path)
    embedder.initialize_db(repo_name)
    return embedder


if __name__ == "__main__":
    main()
