from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as f
from transformers import BertModel


def encode(
        encoder: BertModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, start_tokens, end_tokens
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Encodes inputs with encoder

    :param encoder: bert model
    :param input_ids: LongTensor of (bs, len) shape
    :param attention_mask: BoolTensor of (bs, len) shape
    :param start_tokens:
    :param end_tokens:
    :return: encoded inputs and attention scores
    """

    if input_ids.shape[1] <= 512:
        return process_short_input(encoder, input_ids, attention_mask)
    return process_long_input(encoder, input_ids, attention_mask, start_tokens, end_tokens)


def process_short_input(
        encoder: BertModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Encodes inputs which len is less than 512

    :param encoder: bert model
    :param input_ids: LongTensor of (bs, len) shape
    :param attention_mask: BoolTensor of (bs, len) shape
    :return: encoded inputs and attention scores
    """

    output = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True
    )

    sequence_output = output[0]  # (bs, length, dim)
    attention = output[-1][-1]  # (bs, 12, 512, 512)

    return sequence_output, attention


def process_long_input(
        encoder: BertModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, start_tokens, end_tokens
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Splits the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024

    :param encoder: bert model
    :param input_ids: LongTensor of (bs, len) shape
    :param attention_mask: BoolTensor of (bs, len) shape
    :param start_tokens:
    :param end_tokens:
    :return: encoded inputs and attention scores
    """

    _, length = input_ids.size()
    start_tokens = torch.tensor([start_tokens]).to(input_ids)
    end_tokens = torch.tensor([end_tokens]).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)

    # splits the input to 2 overlapping chunks
    new_input_ids, new_attention_mask, num_seg = [], [], []
    seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()  # a list of bs-th elements
    for i, l_i in enumerate(seq_len):
        if l_i <= 512:
            new_input_ids.append(input_ids[i, :512])
            new_attention_mask.append(attention_mask[i, :512])
            num_seg.append(1)
        else:
            input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
            input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
            attention_mask1 = attention_mask[i, :512]
            attention_mask2 = attention_mask[i, (l_i - 512): l_i]
            new_input_ids.extend([input_ids1, input_ids2])
            new_attention_mask.extend([attention_mask1, attention_mask2])
            num_seg.append(2)

    # runs encoder
    input_ids = torch.stack(new_input_ids, dim=0)  # (x, 512)
    attention_mask = torch.stack(new_attention_mask, dim=0)  # (x, 512)
    output = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )
    sequence_output = output[0]  # (x, 512, dim)
    attention = output[-1][-1]  # (x, 12, 512, 512)

    # restores batch from chunks
    i, new_output, new_attention = 0, [], []
    for (n_s, l_i) in zip(num_seg, seq_len):
        if n_s == 1:
            output = f.pad(sequence_output[i], (0, 0, 0, length - 512))
            att = f.pad(attention[i], (0, length - 512, 0, length - 512))
            new_output.append(output)
            new_attention.append(att)
        elif n_s == 2:
            output1 = sequence_output[i][:512 - len_end]
            mask1 = attention_mask[i][:512 - len_end]
            att1 = attention[i][:, :512 - len_end, :512 - len_end]
            output1 = f.pad(output1, (0, 0, 0, length - 512 + len_end))
            mask1 = f.pad(mask1, (0, length - 512 + len_end))
            att1 = f.pad(att1, (0, length - 512 + len_end, 0, length - 512 + len_end))

            output2 = sequence_output[i + 1][len_start:]
            mask2 = attention_mask[i + 1][len_start:]
            att2 = attention[i + 1][:, len_start:, len_start:]
            output2 = f.pad(output2, (0, 0, l_i - 512 + len_start, length - l_i))
            mask2 = f.pad(mask2, (l_i - 512 + len_start, length - l_i))
            att2 = f.pad(att2, [l_i - 512 + len_start, length - l_i, l_i - 512 + len_start, length - l_i])
            mask = mask1 + mask2 + 1e-10
            output = (output1 + output2) / mask.unsqueeze(-1)
            att = (att1 + att2)
            att = att / (att.sum(-1, keepdim=True) + 1e-10)
            new_output.append(output)
            new_attention.append(att)
        i += n_s

    sequence_output = torch.stack(new_output, dim=0)  # (bs, length, dim)
    attention = torch.stack(new_attention, dim=0)  # (bs, 12, 512, 512)

    return sequence_output, attention
