from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterator

from src.abstract.example.document import Document


class AbstractLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(self, path: Path) -> Iterator[Document]:
        pass
