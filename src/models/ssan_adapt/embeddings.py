import torch
from src.abstract import NO_ENT_IND
from transformers.models.bert.modeling_bert import BertEmbeddings


class NEREmbeddings(torch.nn.Module):
    def __init__(self, prev_embeddings: BertEmbeddings, ner_types_num):
        super().__init__()

        for var in prev_embeddings.__dict__:
            self.__setattr__(var, prev_embeddings.__dict__[var])

        dim = self.word_embeddings.embedding_dim
        self.ner_embeddings = torch.nn.Embedding(num_embeddings=ner_types_num, embedding_dim=dim, padding_idx=NO_ENT_IND)

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
