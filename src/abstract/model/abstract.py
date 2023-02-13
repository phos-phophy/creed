from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

from src.abstract.example import AbstractDataset, Document

from .torch_model import TorchModel


class AbstractModel(TorchModel, metaclass=ABCMeta):

    def __init__(self, relations: Iterable[str]):
        super(AbstractModel, self).__init__()

        self._relations = tuple(['NO_REL'] + list(relations))

        self._no_rel_ind = 0

    @property
    def relations(self):
        return self._relations

    @property
    def no_rel_ind(self):
        return self._no_rel_ind

    @abstractmethod
    def prepare_dataset(self, documents: Iterable[Document], extract_labels: bool = False, evaluation: bool = False) -> AbstractDataset:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass
