from abc import ABCMeta, abstractmethod
from typing import Iterable

from src.abstract.examples import AbstractDataset, Document

from .torch_model import TorchModel


class AbstractModel(TorchModel, metaclass=ABCMeta):

    def __init__(self, entities: Iterable[str], relations: Iterable[str]):
        super(AbstractModel, self).__init__()

        self._entities = tuple(['NO_ENT'] + list(entities))
        self._relations = tuple(['NO_REL'] + list(relations))

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @abstractmethod
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> AbstractDataset:
        pass
