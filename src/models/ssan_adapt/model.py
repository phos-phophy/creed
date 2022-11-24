from typing import Any, Iterable

import torch
from transformers import AutoTokenizer

from src.abstract import AbstractDataset, AbstractModel, Document
from .configure import get_inner_model


class SSANAdaptModel(AbstractModel):

    def __init__(self, tokenizer_path: str, model_type: str, pretrained_model_path: str, hidden_dim: int, dropout: float, **kwargs):
        super(SSANAdaptModel, self).__init__()

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._inner_model = get_inner_model(model_type, pretrained_model_path, **kwargs)

        self._max_seq_len = self._tokenizer.model_max_length

        out_dim = next(module.out_features for module in list(self._inner_model.modules())[::-1] if "out_features" in module.__dict__)

        self._dim_reduction = torch.nn.Linear(out_dim, hidden_dim)
        self._dropout = torch.nn.Dropout(dropout)
        self._dist_emb = torch.nn.Embedding(20, 20, padding_idx=10)
        hidden_dim += 20
        self._bili = torch.nn.Bilinear(hidden_dim, hidden_dim, len(self._model_relations))

    def _forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def _prepare_dataset(self, document: Iterable[Document]) -> AbstractDataset:
        raise NotImplementedError

    def predict(self, document: Document) -> Document:
        pass
