from abc import ABCMeta, abstractmethod
from typing import List

from torch.utils.data import DataLoader

from .abstract import AbstractModel, Document


class AbstractWrapperModel(AbstractModel, metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, output_path: str = None) -> None:
        pass

    @abstractmethod
    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: str = None) -> None:
        pass
