from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, NamedTuple

from src.abstract.examples import AbstractDataset, Document

from .torch_model import TorchModel


class Score(NamedTuple):
    precision: float
    recall: float
    f_score: float


class ModelScore(NamedTuple):
    relations_score: Dict[str, Score]
    macro_score: Score


class AbstractModel(TorchModel, metaclass=ABCMeta):

    def __init__(self, entities: Iterable[str], relations: Iterable[str]):
        super(AbstractModel, self).__init__()

        self._entities = tuple(['NO_ENT'] + list(entities))
        self._relations = tuple(['NO_REL'] + list(relations))

        self._no_ent_ind = 0
        self._no_rel_ind = 0

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @property
    def no_ent_ind(self):
        return self._no_ent_ind

    @property
    def no_rel_ind(self):
        return self._no_rel_ind

    @abstractmethod
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> AbstractDataset:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def score(self, *args, **kwargs) -> ModelScore:
        pass
