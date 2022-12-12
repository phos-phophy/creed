from typing import Iterable

from src.abstract import Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptModel

# from .dataset import WOTypesSSANAdaptDataset


class WOTypesSSANAdaptModel(BaseSSANAdaptModel):
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False):  # -> WOTypesSSANAdaptDataset:
        # return WOTypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation)
        raise NotImplementedError
