import math
from typing import Any, Iterable, Tuple

import torch
from src.abstract import AbstractModel, Document
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertSelfAttention
)

from .dataset import BaseSSANAdaptDataset


class BaseSSANAdaptModel(AbstractModel):
    def __init__(self, relations: Iterable[str], pretrained_model_path: str, tokenizer_path: str):
        super(BaseSSANAdaptModel, self).__init__(relations)

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._model = AutoModel.from_pretrained(pretrained_model_path)
        self._redefine_model_structure()

    def prepare_dataset(self, documents: Iterable[Document], extract_labels=False, evaluation=False) -> BaseSSANAdaptDataset:
        return BaseSSANAdaptDataset(documents, self._tokenizer, extract_labels, evaluation, self.entities, self.relations,
                                    self.no_ent_ind, self.no_rel_ind)

    def forward(
            self,
            input_ids=None,  # (bs, len)
            ner_ids=None,  # (bs, len)
            attention_mask=None,  # (bs, len)
            struct_matrix=None,  # (bs, 5, len, len),
            **kwargs
    ) -> Any:

        struct_matrix = struct_matrix.transpose(0, 1)[:, :, None, :, :]  # (5, bs, 1, len, len)

        self._model.embeddings.ner_ids = ner_ids
        for layer in self._model.encoder.layer:
            layer.attention.self.struct_matrix = struct_matrix

        output = self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        del self._model.embeddings.ner_ids
        for layer in self._model.encoder.layer:
            del layer.attention.self.struct_matrix

        return output

    def compute_loss(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def score(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def _redefine_model_structure(self):
        self._redefine_attention()
        self._redefine_embeddings()

    def _redefine_attention(self):
        for layer in self._model.encoder.layer:
            layer.attention = Attention(layer.attention.self)

    def _redefine_embeddings(self):
        self._model.embeddings = Embeddings(self._model.embeddings, len(self.entities))


# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Embeddings(torch.nn.Module):
    def __init__(self, prev_embeddings: BertEmbeddings, ner_types_num):
        super().__init__()

        for var in prev_embeddings.__dict__:
            self.__setattr__(var, prev_embeddings.__dict__[var])

        dim = self.word_embeddings.embedding_dim
        self.ner_embeddings = torch.nn.Embedding(num_embeddings=ner_types_num, embedding_dim=dim, padding_idx=0)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        ner_embeddings = self.ner_embeddings(self.ner_ids)

        embeddings = inputs_embeds + token_type_embeddings + ner_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Attention(torch.nn.Module):
    def __init__(self, prev_attention: BertSelfAttention):
        super().__init__()

        for var in prev_attention.__dict__:
            self.__setattr__(var, prev_attention.__dict__[var])

        # ================= SSAN-Adapt attention =================
        # 5 struct relations: intra-coref, inter-coref, intra-relate, inter-relate, intra-NA

        tmp = torch.empty(self.num_attention_heads, self.attention_head_size, self.attention_head_size)
        tmp1 = [torch.nn.Parameter(torch.zeros(self.num_attention_heads), requires_grad=True) for _ in range(5)]
        tmp2 = [torch.nn.Parameter(torch.nn.init.xavier_uniform_(tmp), requires_grad=True) for _ in range(5)]

        self.abs_bias = torch.nn.ParameterList(tmp1)
        self.ssan_attention = torch.nn.ParameterList(tmp2)

    def apply_ssan_adapt_attention(self, attention_scores, query, key):
        # b is bs, n is n_heads, q and k are len (length), d is head_size
        # struct_matrix[i] is (bs, 1, len, len)
        for i in range(5):
            attention_bias = torch.einsum("bnqd,ndd,bnkd->bnqk", query, self.ssan_attention[i], key)  # (bs, n_heads, len, len)
            attention_scores += (attention_bias + self.abs_bias[i][None, :, None, None]) * self.struct_matrix[i]  # (bs, n_heads, len, len)

        return attention_scores

    def transpose_for_scores(self, x: torch.Tensor):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,  # (bs, len, dim)
            attention_mask: torch.Tensor = None,
            head_mask: torch.Tensor = None,
            encoder_hidden_states: torch.Tensor = None,  # None
            encoder_attention_mask: torch.Tensor = None,  # None
            past_key_value=None,
            output_attentions: torch.Tensor = False
    ) -> Tuple[torch.Tensor, ...]:

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further, calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further, calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bidirectional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        attention_scores = self.apply_ssan_adapt_attention(attention_scores, query_layer, key_layer)

        # Normalize the attention scores to probabilities.
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
