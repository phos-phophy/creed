from typing import Iterable

from src.abstract import Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptInnerModel

from .dataset import IETypesSSANAdaptDataset


class IETypesSSANAdaptInnerModel(BaseSSANAdaptInnerModel):
    def __init__(self, **kwargs):
        stub_entities = ('NO_ENT', 'ENT')
        super(BaseSSANAdaptInnerModel, self).__init__(entities=stub_entities, **kwargs)

    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> IETypesSSANAdaptDataset:
        return IETypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation, self.stub_entities, self.relations,
                                       self._dist_base, self._dist_ceil)
