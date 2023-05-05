from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from opt_einsum import contract
from src.abstract import AbstractDataset, AbstractModel, CollatedFeatures, DiversifierConfig, Document, PreparedDocument
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModel, AutoTokenizer, BertModel, get_linear_schedule_with_warmup

from .collator import DocUNetCollator
from .datasets import BaseDataset, WOTypesDataset
from .encode import encode
from .unet import UNet


class DocUNet(AbstractModel):
    def __init__(
            self,
            pretrained_model_path: str,
            tokenizer_path: str,
            inner_model_type: str,
            relations: Iterable[str],
            unet_in_dim: int,
            unet_out_dim: int,
            channels: int,
            emb_size: int,
            block_size: int,
            ne: int,
            entities: Optional[Iterable[str]] = None,
    ):
        super().__init__(relations)

        self._entities = tuple(entities) if entities else ()

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._encoder: BertModel = AutoModel.from_pretrained(pretrained_model_path)

        self._linear = torch.nn.Linear(self._encoder.config.hidden_size, unet_in_dim)
        self._unet = UNet(unet_in_dim, unet_out_dim, channels)

        self._head_extractor = torch.nn.Linear(self._encoder.config.hidden_size + unet_out_dim, emb_size)
        self._tail_extractor = torch.nn.Linear(self._encoder.config.hidden_size + unet_out_dim, emb_size)
        self._bilinear = torch.nn.Linear(emb_size * block_size, len(self.relations))

        self._inner_model_type = inner_model_type
        self._ne = ne

    def prepare_dataset(
            self,
            documents: Iterable[Document],
            diversifier: DiversifierConfig,
            desc: str,
            extract_labels: bool = False,
            evaluation: bool = False
    ) -> AbstractDataset:
        if self._inner_model_type == 'base':
            dataset = BaseDataset(documents, self._tokenizer, desc, extract_labels, evaluation, diversifier, self.relations)
        elif self._inner_model_type == 'wo':
            dataset = WOTypesDataset(documents, self._tokenizer, desc, extract_labels, evaluation, diversifier, self.relations)
        else:
            raise ValueError

        return dataset

    def _get_entity_tensors(
            self,
            sequence_output: torch.Tensor,  # (bs, seq_len, dim)
            attention: torch.Tensor,  # (bs, 12, seq_len, len)
            ind: int,
            entity_mentions: List[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns embedding and attention of the given entity

        :param sequence_output: embeddings of the input sequence obtained using the bert encoder
        :param attention: attention obtained using the bert encoder
        :param ind: index of the given entity
        :param entity_mentions: mentions of the given entity

        :return: entity embedding tensor of (dim,) shape and entity attention tensor of (h, seq_len) shape
        """

        offset = 1
        bs, h, _, length = attention.size()

        if len(entity_mentions) == 1:  # there is only one mention
            start, _ = entity_mentions[0]
            if start + offset < length:  # In case the entity mention is truncated due to limited max seq length.
                return sequence_output[ind, start + offset], attention[ind, :, start + offset]
            return torch.zeros(self.config.hidden_size).to(sequence_output), torch.zeros(h, length).to(attention)

        embeddings, attentions = [], []
        for start, _ in entity_mentions:  # iterate over all mentions of the current entity
            if start + offset < length:
                embeddings.append(sequence_output[ind, start + offset])
                attentions.append(attention[ind, :, start + offset])

        # combine mention's vectors to the single one for entity (embedding and attention)
        if len(embeddings) > 0:
            return torch.logsumexp(torch.stack(embeddings, dim=0), dim=0), torch.stack(attentions, dim=0).mean(0)
        return torch.zeros(self.config.hidden_size).to(sequence_output), torch.zeros(h, length).to(attention)

    def _get_ht(
            self,
            sequence_output: torch.Tensor,  # (bs, seq_len, dim)
            attention: torch.Tensor,  # (bs, 12, seq_len, len)
            entity_pos: List[List[List[Tuple[int, int]]]],
            hts: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """ Returns embeddings for head and tail entities and retrieves attentions for whole batch

        :param sequence_output: embeddings of the input sequence obtained using the bert encoder
        :param attention: attention obtained using the bert encoder
        :param entity_pos: positions of the entities in the documents from batch
        :param hts: head & tail entity indexes

        :return: tensor of head entities, tensor of tail entities, _, batch attentions
        """

        hss, tss = [], []
        batch_embeddings = []
        batch_attentions = []

        # iterate over documents in the batch
        for i in range(len(entity_pos)):
            document_embeddings, document_attentions = [], []

            # iterate over entities of the i-th document
            entity_embedding, entity_attention = None, None
            for entity_mentions in entity_pos[i]:

                # entity_embedding is FloatTensor of (dim,) shape
                # entity_attention is FloatTensor of (h, seq_len) shape
                entity_embedding, entity_attention = self._get_entity_tensors(sequence_output, attention, i, entity_mentions)

                document_embeddings.append(entity_embedding)
                document_attentions.append(entity_attention)

            document_attentions.extend([entity_attention] * (self._ne - len(entity_pos[i])))  # ???

            # n_e - number of entities in the document
            document_embeddings = torch.stack(document_embeddings, dim=0)  # (n_e, dim)
            document_attentions = torch.stack(document_attentions, dim=0)  # (n_e, h, seq_len)

            # batch_embeddings.append(document_embeddings)
            batch_attentions.append(document_attentions)

            # r = n_e * (n_e - 1) - number of possible relations in the document
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(document_embeddings, 0, ht_i[:, 0])  # (r, dim)
            ts = torch.index_select(document_embeddings, 0, ht_i[:, 1])  # (r, dim)

            hss.append(hs)
            tss.append(ts)

        # R - number of possible relations in all documents
        hss = torch.cat(hss, dim=0)  # (R, dim)
        tss = torch.cat(tss, dim=0)  # (R, dim)
        return hss, tss, batch_embeddings, batch_attentions

    def _get_channel_map(
            self,
            sequence_output: torch.Tensor,  # (bs, seq_len, dim)
            attentions: List[torch.Tensor]  # a list of tensors with (n_e, 12, seq_len) shape
    ):
        """ Build square attention map for Unet

        :param sequence_output: embeddings of the input sequence obtained using the bert encoder
        :param attentions: list of batch attentions

        :return: square attention matrix of (bs, ne, ne, dim) shape
        """

        index_pair = []
        for i in range(self._ne):
            i_ones = torch.ones((self._ne, 1), dtype=torch.int) * i
            i2j = torch.cat((i_ones, torch.arange(0, self._ne).unsqueeze(1)), dim=-1)  # (ne, 2)
            index_pair.append(i2j)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2).to(sequence_output.device)  # (ne * ne, 2)

        map_rss = []
        for b in range(sequence_output.shape[0]):  # iterate over batch
            document_attentions = attentions[b]  # (n_e, h, seq_len)
            head_attentions = torch.index_select(document_attentions, 0, index_pair[:, 0])  # (ne * ne, h, seq_len)
            tail_attentions = torch.index_select(document_attentions, 0, index_pair[:, 1])  # (ne * ne, h, seq_len)
            ht_attentions = (head_attentions * tail_attentions).mean(1)  # (ne * ne, seq_len)
            ht_attentions = ht_attentions / (ht_attentions.sum(1, keepdim=True) + 1e-5)  # (ne * ne, seq_len)
            rs = contract(sequence_output[b], ht_attentions, subscripts="ld,rl->rd")  # (ne * ne, dim)
            map_rss.append(rs)

        return torch.cat(map_rss, dim=0).reshape(sequence_output.shape[0], self._ne, self._ne, sequence_output.shape[2])

    def forward(
            self,
            input_ids: torch.Tensor,  # (bs, len)
            attention_mask: torch.Tensor,  # (bs, len)
            entity_pos: List[List[List[Tuple[int, int]]]],
            hts: List[torch.Tensor],
            labels: List[torch.Tensor]
    ) -> Any:

        # First stage: Encode inputs via bert encoder
        # sequence_output is FloatTensor of (bs, seq_len, d) shape
        # attention is FloatTensor of (bs, 12, seq_len, seq_len) shape
        sequence_output, attention = encode(
            self._encoder, input_ids, attention_mask, self._tokenizer.cls_token_id, self._tokenizer.sep_token_id
        )

        # Second stage: Extract embeddings for head and tail entities + retrieves attention
        # hs is FloatTensor of (R, dim) shape - head entity embeddings
        # ts is FloatTensor of (R, dim) shape - tail entity embeddings
        # attentions is a list of tensors with (n_e, h, seq_len) shape
        hs, ts, _, attentions = self._get_ht(sequence_output, attention, entity_pos, hts)

        # Third stage: Collect global information between entities via UNet
        feature_map = self._get_channel_map(sequence_output, attentions)  # (bs, ne, ne, dim)
        unet_input = self._linear(feature_map).permute(0, 3, 1, 2).contiguous()  # (bs, u_in_dim, ne, ne)
        unet_output = self._unet(unet_input)  # (bs, ne, ne, u_out_dim)

        # Fourth stage: Combine entity embeddings with information extracted with UNet
        # R - number of possible relations in all documents
        h_t = torch.cat([unet_output[i, hts_i[:, 0], hts_i[:, 1]] for i, hts_i in enumerate(hts)], dim=0)  # (R, u_out_dim)

        hs = torch.tanh(self._head_extractor(torch.cat([hs, h_t], dim=1)))  # (R, e_dim)
        ts = torch.tanh(self._tail_extractor(torch.cat([ts, h_t], dim=1)))  # (R, e_dim)

        # Fifth stage: Classify
        # emb_block = e_dim // block_size
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)  # (R, emb_block, block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)  # (R, emb_block, block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)  # (R, e_dim * block_size)
        logits = self.bilinear(bl)  # (R, class_number)

        output = (self.loss_fnt.get_label(logits, num_labels=len(self.relations)))
        if labels is not None:
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), output)
        return output

    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        pass

    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:
        pass

    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        pass

    def collate_fn(self, documents: List[PreparedDocument]) -> Dict[str, CollatedFeatures]:
        return DocUNetCollator.collate_fn(documents)

    def create_optimizers(self, kwargs: dict) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:

        extract_layer = ["extractor", "bilinear"]
        encoder_layer = ['encoder']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in encoder_layer)], "lr": kwargs["encoder_lr"]},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in extract_layer + encoder_layer)]},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=kwargs["learning_rate"], eps=kwargs["adam_epsilon"])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=kwargs["warmup_steps"], num_training_steps=kwargs["total_steps"]
        )

        return optimizer, scheduler
