from abc import ABCMeta, abstractmethod
from typing import Iterable

from src.abstract.examples import AbstractDataset, Document

from .torch_model import TorchModel


class AbstractModel(TorchModel, metaclass=ABCMeta):
    @abstractmethod
    def prepare_dataset(self, documents: Iterable[Document]) -> AbstractDataset:
        pass
