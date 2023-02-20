from typing import Any, Iterable

from src.abstract import Document
from src.models.ssan_adapt.inner_models.base import BaseSSANAdaptInnerModel

from .dataset import IETypesSSANAdaptDataset


class IETypesSSANAdaptInnerModel(BaseSSANAdaptInnerModel):
    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> IETypesSSANAdaptDataset:
        stub_entities = ()
        return IETypesSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation, stub_entities, self.relations,
                                       self._dist_base, self._dist_ceil)

    def forward(
            self,
            input_ids=None,  # (bs, len)
            attention_mask=None,  # (bs, len)
            struct_matrix=None,  # (bs, 5, len, len),
            **kwargs
    ) -> Any:

        struct_matrix = struct_matrix.transpose(0, 1)[:, :, None, :, :]  # (5, bs, 1, len, len)

        for layer in self._model.encoder.layer:
            layer.attention.self.struct_matrix = struct_matrix

        output = self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        for layer in self._model.encoder.layer:
            del layer.attention.self.struct_matrix

        return output

    def _redefine_model_structure(self):
        self._redefine_attention()
