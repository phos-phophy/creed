from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader

from .abstract import AbstractModel, Document


class AbstractWrapperModel(AbstractModel, metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        """Use for the dev dataset"""
        pass

    @abstractmethod
    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:
        """Use for private test dataset"""
        pass

    @abstractmethod
    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        """Use for the public test dataset"""
        pass
