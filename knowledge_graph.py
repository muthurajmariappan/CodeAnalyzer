"""
Knowledge Graph Module

Manages knowledge graph of repository relationships.
"""

from typing import Dict, Any
import networkx as nx


class KnowledgeGraph:
    """Manages knowledge graph of repository relationships."""

    def __init__(self):
        """Initialize knowledge graph."""
        self.graph = nx.DiGraph()

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of entity (e.g., 'class', 'function', 'file', 'module')
            properties: Properties of the entity
        """
        self.graph.add_node(entity_id, entity_type=entity_type, **properties)

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str,
                         properties: Dict[str, Any] = None):
        """
        Add a relationship between entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relationship_type: Type of relationship (e.g., 'imports', 'calls', 'contains')
            properties: Additional properties of the relationship
        """
        if properties is None:
            properties = {}
        self.graph.add_edge(source_id, target_id, relationship_type=relationship_type, **properties)

    def extract_from_knowledge(self, knowledge: Dict[str, Any], repo_url: str):
        """
        Extract entities and relationships from LLM knowledge output.
        
        Args:
            knowledge: Knowledge dictionary from LLM
            repo_url: Repository URL
        """
        # Add repository as root entity
        repo_id = f"repo:{repo_url}"
        self.add_entity(repo_id, "repository", {"url": repo_url})

        # Add programming languages
        for lang in knowledge.get("programming_languages", []):
            lang_id = f"lang:{lang}"
            self.add_entity(lang_id, "programming_language", {"name": lang})
            self.add_relationship(repo_id, lang_id, "uses")

        # Add frameworks
        for framework in knowledge.get("frameworks", []):
            fw_id = f"framework:{framework}"
            self.add_entity(fw_id, "framework", {"name": framework})
            self.add_relationship(repo_id, fw_id, "uses")

        # Add classes
        for cls in knowledge.get("classes", []):
            cls_id = f"class:{cls.get('name')}"
            file_path = cls.get("file", "unknown")
            file_id = f"file:{file_path}"

            # Add file entity
            self.add_entity(file_id, "file", {"path": file_path})
            self.add_relationship(repo_id, file_id, "contains")

            # Add class entity
            self.add_entity(cls_id, "class", {
                "name": cls.get("name"),
                "description": cls.get("description", ""),
                "file": file_path
            })
            self.add_relationship(file_id, cls_id, "contains")

            # Add methods
            for method in cls.get("methods", []):
                method_id = f"method:{cls.get('name')}.{method}"
                self.add_entity(method_id, "method", {
                    "name": method,
                    "class": cls.get("name")
                })
                self.add_relationship(cls_id, method_id, "contains")

        # Add functions
        for func in knowledge.get("functions", []):
            func_id = f"function:{func.get('name')}"
            file_path = func.get("file", "unknown")
            file_id = f"file:{file_path}"

            # Ensure file exists
            if not self.graph.has_node(file_id):
                self.add_entity(file_id, "file", {"path": file_path})
                self.add_relationship(repo_id, file_id, "contains")

            # Add function entity
            self.add_entity(func_id, "function", {
                "name": func.get("name"),
                "description": func.get("description", ""),
                "file": file_path,
                "parameters": func.get("parameters", [])
            })
            self.add_relationship(file_id, func_id, "contains")

        # Add dependencies
        for dep in knowledge.get("dependencies", []):
            dep_id = f"dependency:{dep}"
            self.add_entity(dep_id, "dependency", {"name": dep})
            self.add_relationship(repo_id, dep_id, "depends_on")

    def to_dict(self) -> Dict[str, Any]:
        """Convert knowledge graph to dictionary format."""
        nodes = []
        edges = []

        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                **data
            })

        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                **data
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "entity_types": {}
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        entity_types = {}
        for _, data in self.graph.nodes(data=True):
            entity_type = data.get("entity_type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": entity_types
        }
