import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type, TypeVar

import torch
from src.abstract.example import AbstractDataset, DiversifierConfig, Document, PreparedDocument
from torch.utils.data import DataLoader

from .collate import CollatedFeatures, Collator

_Model = TypeVar('_Model', bound='AbstractModel')

NO_REL_IND = 0
NO_ENT_IND = 0


class AbstractModel(torch.nn.Module, metaclass=ABCMeta):
    """ An abstract base class that represents relation extraction and classification model """

    def __init__(self, relations: Iterable[str]):
        super(AbstractModel, self).__init__()
        self._relations = tuple(relations)
        self._dummy_param = torch.nn.Parameter(torch.empty(0))

    @property
    def device(self) -> str:
        return self._dummy_param.device

    @property
    def relations(self):
        return self._relations

    def save(self, path: Path, *, rewrite: bool = False) -> None:
        previous_device = self.device
        self.cpu()

        if path.exists() and not rewrite:
            raise Exception(f"Model saving path already exists: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self, f, protocol=4)  # fixed protocol version to avoid issues with serialization on Python 3.6+ versions

        self.to(device=torch.device(previous_device))

    @classmethod
    def load(cls: Type[_Model], path: Path) -> _Model:
        with path.open('rb') as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise Exception(f"Model at {path} is not an instance of {cls}")
        return model

    @abstractmethod
    def prepare_dataset(
            self,
            documents: Iterable[Document],
            diversifier: DiversifierConfig,
            desc: str,
            extract_labels: bool = False,
            evaluation: bool = False
    ) -> AbstractDataset:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        """ Use for the dev dataset """
        pass

    @abstractmethod
    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:
        """ Use for private test dataset """
        pass

    @abstractmethod
    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        """ Use for the public test dataset """
        pass

    def collate_fn(self, documents: List[PreparedDocument]) -> Dict[str, CollatedFeatures]:
        return Collator.collate_fn(documents)
