from abc import ABCMeta, abstractmethod
from typing import TypeVar

from .document import Document
from .torch import TorchModel

_Model = TypeVar('_Model', bound='AbstractModel')


class AbstractModel(TorchModel, metaclass=ABCMeta):
    @abstractmethod
    def predict(self, document: Document) -> Document:
        pass
