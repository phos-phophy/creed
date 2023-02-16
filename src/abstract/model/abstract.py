from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

from src.abstract.example import AbstractDataset, Document

from .torch_model import TorchModel

NO_REL_IND = 0
NO_ENT_IND = 0


class AbstractModel(TorchModel, metaclass=ABCMeta):

    def __init__(self, relations: Iterable[str]):
        super(AbstractModel, self).__init__()
        self._relations = tuple(relations)

    @property
    def relations(self):
        return self._relations

    @abstractmethod
    def prepare_dataset(self, documents: Iterable[Document], extract_labels: bool = False, evaluation: bool = False) -> AbstractDataset:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
