from langchain.tools import tool

from rag_embedder import RAGEmbedder


class Tools:
    def __init__(self,
                 rag: RAGEmbedder):
        self.rag = rag
        self.tools_map = {}

    def tools_definition(self):
        fetch_class_data_schema = {
                "type": "function",
                "function" : {
                    "name": "fetch_class_data",
                    "description": "Fetch data about the given class name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "class_name": {
                                "type": "string",
                                "description": "the class for which details have to be fetched"
                            }
                        },
                        "required": ["class_name"],
                        "additionalProperties": False
                    }
                },
                "strict": True
            }
        fetch_packages_schema = {
            "type": "function",
            "function" : {
                "name": "fetch_packages",
                "description": "Fetch data related to all packages",
                "parameters": {}
            },
            "strict": True
        }
        return [fetch_class_data_schema, fetch_packages_schema]

    def tools(self):
        @tool
        def fetch_class_data(class_name: str):
            """Fetch data about the given class name from a vector store

            Args:
                class_name (str): the class for which details have to be fetched

            Return:
                data about the given class name fetched from a vector store
            """
            print(f"inside fetch_class_data -> class_name({class_name})")
            if not class_name.endswith(".java"):
                class_name = class_name + ".java"
            retrieved_chunks = self.rag.retrieve_relevant_chunks("class", {"class_name": class_name}, n_results=100000)
            print(f"results inside fetch_class_data ({len(retrieved_chunks)})")
            files_text_parts = []
            # Add retrieved chunks
            for chunk in retrieved_chunks[:10]:
                chunk_text = f"=== File: {chunk['metadata'].get('file_path', 'unknown')} (Relevant Chunk) ===\n{chunk['content']}\n\n"
                files_text_parts.append(chunk_text)

            files_text = "".join(files_text_parts)
            print(f"files content {len(files_text)}")
            return files_text

        @tool
        def fetch_packages():
            """Fetch data related to all packages

            Args:


            Return:
                data about all packages fetched from a vector store
            """
            print(f"inside fetch_packages)")
            retrieved_chunks = self.rag.retrieve_relevant_chunks("package", {"is_class": True}, n_results=100000)
            print(f"results inside fetch_packages ({len(retrieved_chunks)})")

            seen_files = set()
            unique_chunks = []
            for chunk in retrieved_chunks:
                file_path = chunk['metadata'].get('file_path', '')
                if file_path not in seen_files and chunk['content'].startswith("package"):
                    unique_chunks.append(chunk)
                    seen_files.add(file_path)

            print(f"results inside fetch_packages after dedup({len(unique_chunks)})")

            files_text_parts = []
            # Add retrieved chunks
            for chunk in unique_chunks:
                text: str = chunk['content']
                chunk_text = f"{text.split(";")[0]};\n"
                files_text_parts.append(chunk_text)

            files_text = "".join(files_text_parts)
            print(f"files content {len(files_text)} {files_text}")
            return files_text

        return [fetch_class_data, fetch_packages]

    def tools_list(self):
        self.tools_map["fetch_class_data"] = self.tools()[0]
        self.tools_map["fetch_packages"] = self.tools()[1]
        return self.tools_map
