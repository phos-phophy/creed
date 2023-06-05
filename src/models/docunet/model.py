import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from opt_einsum import contract
from src.abstract import AbstractDataset, AbstractModel, CollatedFeatures, DiversifierConfig, Document, NO_REL_IND, PreparedDocument
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer, BertModel

from .collator import DocUNetCollator
from .datasets import BaseDataset, IETypesDataset, WOTypesDataset
from .encode import encode
from .loss import ATLoss
from .unet import UNet


class DocUNet(AbstractModel):
    def __init__(
            self,
            pretrained_model_path: str,
            tokenizer_path: str,
            inner_model_type: str,
            relations: Iterable[str],
            num_labels: int,
            unet_in_dim: int,
            unet_out_dim: int,
            channels: int,
            emb_size: int,
            block_size: int,
            ne: int
    ):
        super().__init__(relations)

        self._num_labels = num_labels
        self._unet_in_dim = unet_in_dim
        self._unet_out_dim = unet_out_dim
        self._channels = channels
        self._emb_size = emb_size
        self._block_size = block_size
        self._ne = ne

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._encoder: BertModel = AutoModel.from_pretrained(pretrained_model_path)

        self._linear = torch.nn.Linear(self._encoder.config.hidden_size, unet_in_dim)
        self._unet = UNet(unet_in_dim, unet_out_dim, channels)

        self._head_extractor = torch.nn.Linear(self._encoder.config.hidden_size + unet_out_dim, emb_size)
        self._tail_extractor = torch.nn.Linear(self._encoder.config.hidden_size + unet_out_dim, emb_size)
        self._bilinear = torch.nn.Linear(emb_size * block_size, len(self.relations))

        self._inner_model_type = inner_model_type

        self._loss_fnt = ATLoss()

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
        elif self._inner_model_type == 'ie':
            dataset = IETypesDataset(documents, self._tokenizer, desc, extract_labels, evaluation, diversifier, self.relations)
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

        offset = 1  # shift because there is a special token
        bs, h, _, length = attention.size()

        if len(entity_mentions) == 1:  # there is only one mention
            start, _ = entity_mentions[0]
            if start + offset < length:  # In case the entity mention is truncated due to limited max seq length.
                return sequence_output[ind, start + offset], attention[ind, :, start + offset]
            return torch.zeros(self._encoder.config.hidden_size).to(sequence_output), torch.zeros(h, length).to(attention)

        embeddings, attentions = [], []
        for start, _ in entity_mentions:  # iterate over all mentions of the current entity
            if start + offset < length:
                embeddings.append(sequence_output[ind, start + offset])
                attentions.append(attention[ind, :, start + offset])

        # combine mention's vectors to the single one for entity (embedding and attention)
        if len(embeddings) > 0:
            return torch.logsumexp(torch.stack(embeddings, dim=0), dim=0), torch.stack(attentions, dim=0).mean(0)
        return torch.zeros(self._encoder.config.hidden_size).to(sequence_output), torch.zeros(h, length).to(attention)

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

            # expand document_attentions to at least ne elements
            document_attentions.extend([entity_attention] * (self._ne - len(entity_pos[i])))  # list of ne elements

            # n_e - number of entities in the document
            document_embeddings = torch.stack(document_embeddings, dim=0)  # (ne, dim)
            document_attentions = torch.stack(document_attentions, dim=0)  # (ne, h, seq_len)

            # batch_embeddings.append(document_embeddings)
            batch_attentions.append(document_attentions)

            # r = n_e * (n_e - 1) - number of possible relations in the document
            ht_i = hts[i].to(sequence_output.device)
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
            document_attentions = attentions[b]  # (ne, h, seq_len)
            head_attentions = torch.index_select(document_attentions, 0, index_pair[:, 0])  # (ne * ne, h, seq_len)
            tail_attentions = torch.index_select(document_attentions, 0, index_pair[:, 1])  # (ne * ne, h, seq_len)
            ht_attentions = (head_attentions * tail_attentions).mean(1)  # (ne * ne, seq_len)
            ht_attentions = ht_attentions / (ht_attentions.sum(1, keepdim=True) + 1e-5)  # (ne * ne, seq_len)
            rs = contract("ld,rl->rd", sequence_output[b], ht_attentions)  # (ne * ne, dim)
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
        # attentions is a list of tensors with (ne, h, seq_len) shape
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
        b1 = hs.view(-1, self._emb_size // self._block_size, self._block_size)  # (R, emb_block, block_size)
        b2 = ts.view(-1, self._emb_size // self._block_size, self._block_size)  # (R, emb_block, block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self._emb_size * self._block_size)  # (R, e_dim * block_size)
        logits = self._bilinear(bl)  # (R, class_number)

        output = self._loss_fnt.get_label(logits, num_labels=self._num_labels)  # (R, class_number)
        if labels is not None:
            labels = torch.cat(labels, dim=0).to(logits)  # (R, class_number)
            loss = self._loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output), output)
        return output

    def evaluate(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Evaluating')

    def predict(self, documents: List[Document], dataloader: DataLoader, output_path: Path) -> None:

        preds, htss = [], []
        for inputs in tqdm(dataloader, desc='Predicting'):
            self.eval()

            inputs.update({
                'input_ids': inputs["input_ids"].cuda() if torch.cuda.is_available() else inputs["input_ids"],
                'attention_mask': inputs["attention_mask"].cuda() if torch.cuda.is_available() else inputs["attention_mask"],
            })

            with torch.no_grad():
                outputs = self(**inputs)
                if isinstance(outputs, tuple):
                    _, pred = outputs
                else:
                    pred = outputs  # (R, class_number)

                pred = pred.cpu().numpy()  # (R, class_number)
                pred[np.isnan(pred)] = 0

            htss += inputs["hts"]
            preds.append(pred)

        preds = np.concatenate(preds, axis=0).astype(np.float32)  # (R_total, class_number)

        h_idx, t_idx, title = [], [], []
        for hts, d in zip(htss, documents):
            h_idx += hts[:, 0].tolist()
            t_idx += hts[:, 1].tolist()
            title += [d.doc_id for _ in hts]

        output_preds = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
            for p in pred:
                if p != NO_REL_IND:
                    output_preds.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': self.relations[p]
                        }
                    )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as file:
            json.dump(output_preds, file, indent=4)

    def test(self, dataloader: DataLoader, output_path: Path = None) -> None:
        self._evaluate(dataloader, output_path, 'Test')

    def _evaluate(self, dataloader: DataLoader, output_path: Path = None, desc: str = None) -> None:
        if Path is None:
            pass

        total_loss = 0.0
        labels_ids, preds = [], []
        for inputs in tqdm(dataloader, desc=desc):
            self.eval()

            inputs.update({
                'input_ids': inputs["input_ids"].cuda() if torch.cuda.is_available() else inputs["input_ids"],
                'attention_mask': inputs["attention_mask"].cuda() if torch.cuda.is_available() else inputs["attention_mask"],
            })

            with torch.no_grad():
                outputs = self(**inputs)
                if isinstance(outputs, tuple):
                    loss, pred = outputs
                    total_loss += loss.item()
                else:
                    pred = outputs  # (R, class_number)

                labels = [i.cpu().numpy() for i in inputs["labels"]]  # (R, class_number)
                pred = pred.cpu().numpy()  # (R, class_number)
                pred[np.isnan(pred)] = 0

            preds.append(pred)
            labels_ids += labels

        preds = np.concatenate(preds, axis=0).astype(np.float32)  # (R_total, class_number)
        labels_ids = np.concatenate(labels_ids, axis=0).astype(np.float32)  # (R_total, class_number)

        mask = np.ones_like(labels_ids, dtype=np.bool)  # (R_total, class_number)
        mask[:, NO_REL_IND] = False  # (R_total, class_number)

        total_labels = labels_ids[mask].sum()
        total_preds = preds[mask].sum()

        correct_preds = (preds == labels_ids)[mask & (labels_ids == 1)].sum()

        precision = correct_preds / total_preds
        recall = correct_preds / total_labels
        f = (2 * recall * precision / (recall + precision + 1e-20))

        result = {
            "loss": float(loss),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f)
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as file:
            json.dump(result, file, indent=4)

    def collate_fn(self, documents: List[PreparedDocument]) -> Dict[str, CollatedFeatures]:
        return DocUNetCollator.collate_fn(documents)

    def create_optimizer(self, kwargs: dict) -> torch.optim.Optimizer:

        extract_layer = ["extractor", "bilinear"]
        encoder_layer = ['encoder']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in encoder_layer)], "lr": kwargs["encoder_lr"]},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in extract_layer)], "lr": 1e-4},
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in extract_layer + encoder_layer)]},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=kwargs["learning_rate"], eps=kwargs["adam_epsilon"])

        return optimizer
