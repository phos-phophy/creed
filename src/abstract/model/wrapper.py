from abc import ABCMeta, abstractmethod

from src.abstract import AbstractDataset

from .abstract import AbstractModel


class AbstractWrapperModel(AbstractModel, metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, dataset: AbstractDataset, output_path: str = None):
        pass

    @abstractmethod
    def predict(self, dataset: AbstractDataset, output_path: str = None):
        pass
