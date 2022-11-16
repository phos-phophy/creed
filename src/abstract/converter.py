from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Iterable

from .document import Document


class AbstractConverter(metaclass=ABCMeta):

    @abstractmethod
    def convert(self, path: Path) -> Iterable[Document]:
        pass
