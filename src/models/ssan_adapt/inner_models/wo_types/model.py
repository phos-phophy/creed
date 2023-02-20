from typing import Iterable

from src.abstract import Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptInnerModel

from .dataset import WOTypesSSANAdaptDataset


class WOTypesSSANAdaptInnerModel(BaseSSANAdaptInnerModel):
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False):  # -> WOTypesSSANAdaptDataset:
        stub_entities = ()
        return WOTypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation, stub_entities, self.relations,
                                       self._dist_base, self._dist_ceil)
