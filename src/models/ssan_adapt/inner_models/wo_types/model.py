from pathlib import Path
from typing import Iterable

from src.abstract import DiversifierConfig, Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptInnerModel

from .dataset import WOTypesSSANAdaptDataset


class WOTypesSSANAdaptInnerModel(BaseSSANAdaptInnerModel):

    def __init__(self, **kwargs):
        stub_entities = ('NO_ENT', 'ENT')
        super(WOTypesSSANAdaptInnerModel, self).__init__(entities=stub_entities, **kwargs)

    def prepare_dataset(
            self,
            documents: Iterable[Document],
            diversifier: DiversifierConfig,
            desc: str,
            extract_labels=False,
            evaluation=False,
            cache_dir: Path = None,
            dataset_name: str = ''
    ) -> WOTypesSSANAdaptDataset:

        if diversifier.active:
            raise ValueError("WO SSAN Adapt model and active diversifier are not compatible!")

        return WOTypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation, self.entities, self.relations,
                                       self._dist_base, self._dist_ceil, desc, diversifier, cache_dir, dataset_name)
