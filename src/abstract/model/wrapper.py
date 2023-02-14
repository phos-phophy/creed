from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader

from .abstract import AbstractModel


class AbstractWrapperModel(AbstractModel, metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, dataloader: DataLoader, output_path: str = None) -> None:
        pass

    @abstractmethod
    def predict(self, dataloader: DataLoader, output_path: str = None) -> None:
        pass
