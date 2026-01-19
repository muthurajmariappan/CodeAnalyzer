import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
from rag_embedder import RAGEmbedder


class GitRAGLoader:

    def __init__(self, repo_url: str, repo_name: str, max_files: int, rag: RAGEmbedder):
        self.repo_url = repo_url
        self.repo_name = repo_name
        self.max_files = max_files
        self.temp_dir = None
        self.rag = rag

    def clone_repository(self) -> Path:
        """
        Clone a GitHub repository to a temporary directory.

        Args:


        Returns:
            Path to the cloned repository
        """
        # Parse the repository URL
        parsed = urlparse(self.repo_url)
        if not parsed.netloc or "github.com" not in parsed.netloc:
            raise ValueError(f"Invalid GitHub URL: {self.repo_url}")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="github_repo_")
        repo_path = Path(self.temp_dir)

        print(f"Cloning repository to {repo_path}...")
        try:
            # Clone the repository
            result = subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(repo_path)],
                capture_output=True,
                text=True,
                check=True
            )
            print("Repository cloned successfully.")

            return repo_path
        except subprocess.CalledProcessError as e:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            raise RuntimeError(f"Failed to clone repository: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError(
                "Git is not installed. Please install Git to clone repositories."
            )

    def read_repository_files(self, repo_path: Path) -> Dict[str, str]:
        """
        Read files from the repository, excluding common ignore patterns.

        Args:
            repo_path: Path to the repository

        Returns:
            Dictionary mapping file paths to their contents
        """
        files_content = {}
        file_count = 0

        # Common ignore patterns
        ignore_patterns = {
            ".git", ".github", "__pycache__", "node_modules", ".venv", "venv",
            "env", ".env", "dist", ".editorconfig", ".pytest_cache", ".mypy_cache",
            ".idea", ".vscode", "*.pyc", "*.pyo", "*.pyd", ".DS_Store",
            "*.egg-info", ".coverage", "htmlcov", ".tox", ".chroma"
        }

        # Common file extensions to include
        code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
            ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
            ".scala", ".clj", ".sh", ".yaml", ".yml", ".json", ".xml",
            ".html", ".css", ".scss", ".md", ".txt", ".dockerfile", ".sql",
            ".kts", ".properties"
        }

        print(f"Reading files from repository (max {self.max_files} files)...")

        for root, dirs, filenames in os.walk(repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(
                pattern in d.lower() for pattern in ignore_patterns
            )]
            print(f"{len(filenames)} {filenames} files discovered")
            for filename in filenames:
                if file_count >= self.max_files:
                    print(f"Reached maximum file limit ({self.max_files}). Stopping for {filename}.")
                    break

                file_path = Path(root) / filename
                relative_path = file_path.relative_to(repo_path)

                # Skip ignored files
                if any(pattern in str(relative_path).lower() for pattern in ignore_patterns):
                    continue

                # Include code files and important config files
                if file_path.suffix.lower() in code_extensions or filename.lower() in {
                    "dockerfile", "makefile", ".gitignore", ".dockerignore",
                    "requirements.txt", "package.json", "pom.xml", "build.gradle"
                }:
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            # Limit file size to avoid token limits
                            if len(content) > 100000:  # ~100KB
                                content = content[:100000] + "\n... (truncated)"
                            files_content[str(relative_path)] = content
                            print(f"{relative_path} with size {len(content)} added to files")
                            file_count += 1
                    except Exception as e:
                        print(f"Warning: Could not read {relative_path}: {e}")

        print(f"Read {len(files_content)} files from repository.")
        return files_content

    def embed_repository(self, files_content: Dict[str, str]):
        """
        Embed all repository files using RAG.

        Args:
            files_content: Dictionary mapping file paths to contents

        """
        print("Embedding repository files...")
        total_files = len(files_content)

        for i, (file_path, content) in enumerate(files_content.items(), 1):
            # if i % 10 == 0:
            print(f"Embedding progress: {i}/{total_files} files...")
            try:
                if not file_path.endswith("sql"):
                    print(f"embedding {file_path}")
                    self.rag.embed_and_store(file_path, content)
            except Exception as e:
                print(f"Warning: Failed to embed {file_path}: {e}")
                traceback.print_exc()

        print("Repository embedding complete!")

    def load_files_to_rag(self) -> Dict[str, str]:
        # Clone repository
        repo_path = self.clone_repository()

        # Read files
        files_content = self.read_repository_files(repo_path)

        if not files_content:
            raise ValueError("No files found in repository.")

        # Embed repository using RAG
        # self.embed_repository(files_content)

        return files_content

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
