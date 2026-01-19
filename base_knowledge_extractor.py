from abc import ABC, abstractmethod
from typing import Dict


class KnowledgeExtractor(ABC):

    @abstractmethod
    def extract(self, files_content: Dict[str, str]) -> Dict:
        pass
