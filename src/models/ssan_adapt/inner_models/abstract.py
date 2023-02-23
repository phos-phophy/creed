import math
from abc import ABCMeta
from typing import Iterable


from src.abstract import AbstractModel, get_tokenizer_len_attribute
from transformers import AutoModel, AutoTokenizer


class AbstractSSANAdaptInnerModel(AbstractModel, metaclass=ABCMeta):

    def __init__(
            self,
            entities: Iterable[str],
            relations: Iterable[str],
            pretrained_model_path: str,
            tokenizer_path: str,
            dist_base: int
    ):
        super(AbstractSSANAdaptInnerModel, self).__init__(relations)

        self._entities = tuple(entities)

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._model = AutoModel.from_pretrained(pretrained_model_path)

        self._dist_base = dist_base

        len_attr = get_tokenizer_len_attribute(self._tokenizer)

        self._dist_ceil = math.ceil(math.log(self._tokenizer.__getattribute__(len_attr), self._dist_base)) + 1
        self._dist_emb_dim = self._dist_ceil * 2

    @property
    def entities(self):
        return self._entities

    @property
    def dist_ceil(self):
        return self._dist_ceil

    @property
    def config(self):
        return self._model.config
