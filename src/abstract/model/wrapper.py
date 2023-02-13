from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

from src.abstract.example import Document

from .abstract import AbstractModel


class AbstractWrapperModel(AbstractModel, metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, documents: Iterable[Document], output_path: str = None) -> Any:
        pass

    @abstractmethod
    def predict(self, documents: Iterable[Document], output_path: str = None) -> Any:
        pass
