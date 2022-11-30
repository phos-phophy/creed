from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

from src.abstract import AbstractDataset, Document, TorchModel


class AbstractInnerModel(TorchModel, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def prepare_dataset(self, document: Iterable[Document]) -> AbstractDataset:
        pass
