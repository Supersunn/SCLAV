import torch

from .base import BaseLoss
from ..builder import LOSSES

@LOSSES.register()
class TopKLoss(BaseLoss):

    def __init__(self, q: int = 16, use_all_negative: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.q = q
        self.use_all_negative = use_all_negative

    def forward(self, outputs, labels, seq_len):
        prob = outputs['prob'].flatten(1)  # BxT
        return self.calculate_mil_loss(prob, labels, seq_len)

    def calculate_mil_loss(self, prob, labels, seq_len):
        loss = torch.tensor(0.0).to(prob.device)
        for i in range(prob.shape[0]):
            lens = seq_len[i]
            tmp1 = prob[i, :lens]

            if labels[i] == 0 and self.use_all_negative:
                k = lens.item()
            else:
                k = (lens.item() + self.q - 1) // self.q

            label = labels[i].repeat(k)

            tmp, _ = torch.topk(tmp1, k=k, largest=True)
            loss += self.loss(tmp, label)
        return loss / prob.shape[0]
