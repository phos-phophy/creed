from typing import Any, Iterable

from src.abstract import Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptInnerModel

# from .dataset import WOTypesSSANAdaptDataset


class WOTypesSSANAdaptInnerModel(BaseSSANAdaptInnerModel):
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False):  # -> WOTypesSSANAdaptDataset:
        # return WOTypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation)
        raise NotImplementedError

    def compute_loss(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def score(self, *args, **kwargs) -> Any:
        raise NotImplementedError
