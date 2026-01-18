# GitHub Repository Knowledge Extractor with RAG and Knowledge Graph

A Python program that analyzes GitHub repositories using **RAG (Retrieval-Augmented Generation)**, **Knowledge Graphs**, and **Large Language Models (LLMs)** to extract structured knowledge about codebases. The program is optimized to respect LLM token limits while providing comprehensive analysis.

## Features

- **Repository Cloning**: Automatically clones GitHub repositories for analysis
- **RAG (Retrieval-Augmented Generation) using LangChain**: 
  - Embeds repository contents using OpenAI embeddings via LangChain
  - Uses ChromaDB vector store through LangChain
  - Uses LangChain's RecursiveCharacterTextSplitter for intelligent chunking
  - Retrieves relevant code sections using LangChain's similarity search
- **Knowledge Graph**: 
  - Builds a graph of relationships between code entities
  - Stores relationships between classes, functions, files, dependencies, etc.
  - Uses NetworkX for graph management
- **Token Optimization**: 
  - Automatically counts tokens using tiktoken
  - Optimizes content to fit within model token limits
  - Supports different LLM models with appropriate token limits
- **Smart File Filtering**: Excludes common build artifacts, dependencies, and cache files
- **LLM-Powered Analysis using LangChain**: Uses LangChain's LLM abstractions to work with multiple providers
- **Multiple LLM Providers via LangChain**: 
  - **OpenAI**: Use GPT-4, GPT-3.5, and other OpenAI models via `langchain-openai`
  - **Ollama**: Use open source models like Llama, Mistral, etc. via `langchain-community`
- **Comprehensive Extraction**: Identifies:
  - Programming languages used
  - Frameworks and libraries
  - Classes and their methods
  - Functions and their parameters
  - High-level architecture and design patterns
  - Key dependencies
  - Overall project summary

## Prerequisites

1. **Python 3.8+**
2. **Git** installed and available in PATH
3. **LLM Provider** (choose one):
   - **OpenAI**: API key from [OpenAI](https://platform.openai.com/api-keys) (for OpenAI models)
   - **Ollama**: Install [Ollama](https://ollama.ai/) and pull a model (for open source models)

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:
- `openai` - For OpenAI LLM and embeddings API (optional if using Ollama)
- `tiktoken` - For token counting
- `chromadb` - For vector database (RAG, optional)
- `networkx` - For knowledge graph
- `requests` - For Ollama API communication (optional if using OpenAI)
- `langchain` - LangChain core framework
- `langchain-openai` - LangChain OpenAI integration
- `langchain-community` - LangChain community integrations (Ollama, etc.)
- `langchain-chroma` - LangChain ChromaDB integration

## Configuration

### For OpenAI Provider

Set your OpenAI API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or you can pass it directly via the `--api-key` parameter (see Usage below).

### For Ollama Provider

1. Install [Ollama](https://ollama.ai/)
2. Pull a model (e.g., `ollama pull llama3.2`)
3. Ensure Ollama is running (default: http://localhost:11434)

## Usage

### Basic Usage

```bash
python github_repo_analyzer.py https://github.com/user/repository
```

### Advanced Usage

#### OpenAI Provider

```bash
# Specify API key directly
python github_repo_analyzer.py https://github.com/user/repository --provider openai --api-key your-key

# Use a different OpenAI model
python github_repo_analyzer.py https://github.com/user/repository --provider openai --model gpt-4o

# Use a different embedding model
python github_repo_analyzer.py https://github.com/user/repository --provider openai --embedding-model text-embedding-3-large
```

#### Ollama Provider (Open Source)

```bash
# Use Ollama with default model (llama3.2)
python github_repo_analyzer.py https://github.com/user/repository --provider ollama

# Use a specific Ollama model
python github_repo_analyzer.py https://github.com/user/repository --provider ollama --model llama3.1

# Use custom Ollama URL
python github_repo_analyzer.py https://github.com/user/repository --provider ollama --ollama-url http://localhost:11434
```

#### Common Options

```bash
# Analyze more files (default is 100)
python github_repo_analyzer.py https://github.com/user/repository --max-files 200

# Disable RAG (use direct file analysis only)
python github_repo_analyzer.py https://github.com/user/repository --no-rag

# Save output to a file
python github_repo_analyzer.py https://github.com/user/repository --output results.json
```

### Command Line Arguments

- `repo_url` (required): GitHub repository URL
- `--provider`: LLM provider to use (default: `openai`)
  - Options: `openai`, `ollama`
- `--model`: LLM model to use
  - OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo` (default: `gpt-4o-mini`)
  - Ollama: Any model available in Ollama (default: `llama3.2`)
- `--api-key`: OpenAI API key (required for OpenAI provider, optional if OPENAI_API_KEY env var is set)
- `--ollama-url`: Ollama API base URL (default: `http://localhost:11434`)
- `--embedding-model`: Embedding model for RAG (default: `text-embedding-3-small`, OpenAI only)
  - Options: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`
- `--max-files`: Maximum number of files to analyze (default: 100)
- `--no-rag`: Disable RAG (embedding and retrieval) - uses direct file analysis
- `--output`: Output JSON file path (optional, prints to stdout if not specified)

## How It Works

1. **Repository Cloning**: Clones the GitHub repository to a temporary directory
2. **File Reading**: Reads and filters relevant code files (excludes build artifacts, node_modules, etc.)
3. **RAG Embedding using LangChain** (if enabled):
   - Chunks file contents using LangChain's RecursiveCharacterTextSplitter
   - Generates embeddings using LangChain's OpenAIEmbeddings
   - Stores embeddings in ChromaDB vector store via LangChain
4. **Semantic Retrieval using LangChain**:
   - Uses LangChain's similarity search to retrieve relevant code sections
   - Combines retrieved chunks with key files (README, main files, etc.)
5. **Token Optimization**:
   - Counts tokens using tiktoken
   - Optimizes content to fit within the model's token limit
   - Reserves tokens for prompt and response
6. **LLM Analysis using LangChain**:
   - Sends optimized content to the LLM via LangChain's LLM interface
   - Extracts structured knowledge about the codebase
7. **Knowledge Graph Construction**:
   - Builds a graph of relationships between entities
   - Stores relationships: classes, functions, files, dependencies, etc.
8. **Output**: Returns comprehensive analysis with knowledge graph

## Output Format

The program outputs a JSON structure containing:

```json
{
  "programming_languages": ["Python", "JavaScript"],
  "frameworks": ["Flask", "React"],
  "classes": [
    {
      "name": "ClassName",
      "file": "path/to/file.py",
      "description": "Brief description",
      "methods": ["method1", "method2"]
    }
  ],
  "functions": [
    {
      "name": "function_name",
      "file": "path/to/file.py",
      "description": "Brief description",
      "parameters": ["param1", "param2"]
    }
  ],
  "high_level_design": {
    "architecture": "Description of overall architecture",
    "patterns": ["MVC", "Singleton"],
    "key_components": ["component1", "component2"],
    "data_flow": "Description of how data flows through the system"
  },
  "dependencies": ["flask", "react", "numpy"],
  "summary": "High-level summary of what this repository does",
  "knowledge_graph": {
    "nodes": [
      {
        "id": "repo:https://github.com/user/repo",
        "entity_type": "repository",
        "url": "https://github.com/user/repo"
      },
      {
        "id": "class:MyClass",
        "entity_type": "class",
        "name": "MyClass",
        "description": "...",
        "file": "src/main.py"
      }
    ],
    "edges": [
      {
        "source": "repo:https://github.com/user/repo",
        "target": "class:MyClass",
        "relationship_type": "contains"
      }
    ],
    "statistics": {
      "total_nodes": 50,
      "total_edges": 75,
      "entity_types": {
        "repository": 1,
        "class": 10,
        "function": 20,
        "file": 15,
        "dependency": 4
      }
    }
  }
}
```

## Token Optimization

The program automatically optimizes content to respect LLM token limits:

- **Token Counting**: Uses tiktoken to accurately count tokens for the selected model
- **Model Limits**: Automatically detects and respects model-specific token limits:
  - `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`: 128,000 tokens
  - `gpt-4`: 8,192 tokens
  - `gpt-3.5-turbo`: 16,385 tokens
- **Smart Truncation**: Intelligently truncates content while preserving important information
- **Reserved Tokens**: Reserves tokens for prompt structure and LLM response

## Examples

### Using OpenAI

```bash
python github_repo_analyzer.py https://github.com/octocat/Hello-World --provider openai --output analysis.json
```

### Using Ollama (Open Source)

```bash
# Make sure Ollama is running and you have a model pulled
ollama pull llama3.2

# Run the analyzer
python github_repo_analyzer.py https://github.com/octocat/Hello-World --provider ollama --output analysis.json
```

This will:
1. Clone the repository to a temporary directory
2. Read and filter relevant code files
3. Embed files using RAG (if enabled)
4. Retrieve relevant code sections using semantic search
5. Optimize content to fit token limits
6. Send optimized content to the LLM for analysis
7. Build a knowledge graph of relationships
8. Extract structured knowledge
9. Save results to `analysis.json`
10. Clean up temporary files

## Limitations

- **File Limit**: By default, analyzes up to 100 files to manage token costs and API limits
- **File Size**: Large files (>100KB) are truncated
- **Token Limits**: Very large repositories may require adjusting `--max-files` or using models with higher token limits
- **API Costs**: Uses OpenAI API which incurs costs based on usage (both for LLM and embeddings)
- **Embedding Costs**: RAG requires embedding all files, which adds to API costs

## Troubleshooting

### "Git is not installed"
Install Git from [https://git-scm.com/](https://git-scm.com/)

### "OpenAI API key not provided"
- For OpenAI provider: Set the `OPENAI_API_KEY` environment variable or use the `--api-key` parameter
- For Ollama provider: Ensure Ollama is running and accessible at the specified URL

### "Failed to clone repository"
- Check that the repository URL is correct and publicly accessible
- Ensure you have internet connectivity
- Verify Git is properly installed

### "Required package not installed"
Run `pip install -r requirements.txt` to install all dependencies

### "LangChain is not installed"
- Install LangChain packages: `pip install langchain langchain-openai langchain-community langchain-chroma`
- Or install all dependencies: `pip install -r requirements.txt`

### High API costs (OpenAI)
- Use `--provider ollama` to use free open source models locally
- Use `--no-rag` to disable RAG and reduce embedding costs
- Reduce `--max-files` to analyze fewer files
- Use `gpt-4o-mini` instead of `gpt-4o` for lower LLM costs

### Ollama connection issues
- Ensure Ollama is installed and running: `ollama serve`
- Check that the model is available: `ollama list`
- Verify the Ollama URL is correct (default: http://localhost:11434)

## Architecture

The program consists of several key components built on LangChain:

- **LLMProvider (LangChain-based)**: Abstract interface for LLM providers
  - **OpenAIProvider**: Uses `langchain-openai.ChatOpenAI` for OpenAI models
  - **OllamaProvider**: Uses `langchain-community.llms.Ollama` for open source models
- **TokenCounter**: Handles token counting and content optimization
- **RAGEmbedder (LangChain-based)**: Manages RAG using LangChain components
  - Uses `langchain_openai.OpenAIEmbeddings` for embeddings
  - Uses `langchain_community.vectorstores.Chroma` for vector storage
  - Uses `langchain.text_splitter.RecursiveCharacterTextSplitter` for chunking
- **KnowledgeGraph**: Builds and manages the knowledge graph
- **GitHubRepoAnalyzer**: Main orchestrator that coordinates all components

## License

This project is provided as-is for educational and development purposes.
