import torch
from src.abstract import NO_REL_IND


def multilabel_categorical_crossentropy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """ Computes MCCE

    :param y_true: gold labels; FloatTensor of (R, class_number) shape
    :param y_pred: predicted logits; FloatTensor of (R, class_number) shape
    :return: FloatTensor of (R,) shape
    """

    y_pred = (1 - 2 * y_true) * y_pred  # (R, class_number)
    y_pred_neg = y_pred - y_true * 1e30  # (R, class_number)
    y_pred_pos = y_pred - (1 - y_true) * 1e30  # (R, class_number)
    zeros = torch.zeros_like(y_pred[..., :1])  # (R, 1)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)  # (R, class_number + 1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)  # (R, class_number + 1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)  # (R,)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)  # (R,)
    return neg_loss + pos_loss


class ATLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ Computes loss

        :param logits: predicted logits; FloatTensor of (R, class_number) shape
        :param labels: gold labels; FloatTensor of (R, class_number) shape
        :return: loss
        """

        loss = multilabel_categorical_crossentropy(labels, logits)
        loss = loss.mean()
        return loss

    @staticmethod
    def get_label(logits: torch.Tensor, num_labels: int = 0) -> torch.Tensor:
        """ Returns predicted labels

        :param logits: FloatTensor of (R, num_classes) shape
        :param num_labels: number of desired labels
        :return: Tensor of (R, num_classes) shape
        """

        th_logit = torch.zeros_like(logits[..., :1])  # (R, 1)
        output = torch.zeros_like(logits).to(logits)  # (R, num_classes)
        mask = (logits > th_logit)  # (R, num_classes)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)  # (R, num_labels)
            top_v = top_v[:, -1]  # (R,)
            mask = (logits >= top_v.unsqueeze(1)) & mask  # (R, num_classes)
        output[mask] = 1.0
        output[:, NO_REL_IND] = ((output.sum(1) - output[:, NO_REL_IND]) == 0.).to(logits)

        return output
