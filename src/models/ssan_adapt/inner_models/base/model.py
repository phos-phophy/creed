from typing import Any, Iterable

from src.abstract import AbstractModel, Document
from transformers import AutoModel, AutoTokenizer

from .dataset import BaseSSANAdaptDataset


class BaseSSANAdaptModel(AbstractModel):
    def __init__(self, entities: Iterable[str], relations: Iterable[str], pretrained_model_path: str, tokenizer_path: str, **kwargs):
        super(BaseSSANAdaptModel, self).__init__(entities, relations)

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._model = self._change_model(AutoModel.from_pretrained(pretrained_model_path))

    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> BaseSSANAdaptDataset:
        return BaseSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation)

    def forward(
            self,
            input_ids=None,  # (bs, len)
            ner_ids=None,  # (bs, len)
            attention_mask=None,  # (bs, len)
            struct_mask=None  # (bs, 5, len, len)
    ) -> Any:
        return self._model(input_ids=input_ids, ner_ids=ner_ids, attention_mask=attention_mask, struct_mask=struct_mask)

    def _change_model(self, model):
        return model
