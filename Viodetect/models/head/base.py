from abc import ABCMeta, abstractmethod

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

import torch


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class BaseHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__()
        self.mil_type = kwargs.get('mil_type', 'mean')
        self.mean_type = kwargs.get('mean_type', 'logits')

    def init_weights(self):
        self.apply(weight_init)

    def mil(self, logits, seq_len):
        logits = logits.squeeze(-1)
        v_logits = torch.zeros(0).to(logits.device)
        if self.mean_type == 'prob':
            logits = torch.sigmoid(logits)
        lens = len(logits)
        for i in range(lens):
            if self.mil_type == 'mean':
                tmp = torch.mean(logits[i, :seq_len[i]]).view(1)
            elif self.mil_type == 'topk':
                tmp, _ = torch.topk(
                    logits[i][:seq_len[i]],
                    k=torch.div(seq_len[i], 16, rounding_mode='floor') + 1,
                    largest=True)
                tmp = torch.mean(tmp).view(1)
            else:
                raise NotImplementedError()
            v_logits = torch.cat((v_logits, tmp))
        return v_logits

    def mil_fro_prob(self, prob, seq_len):
        prob = prob.squeeze(-1)
        v_probs = torch.zeros(0).to(prob.device)
        lens = len(prob)
        for i in range(lens):
            if self.mil_type == 'mean':
                tmp = torch.mean(prob[i, :seq_len[i]]).view(1)
            elif self.mil_type == 'topk':
                tmp, _ = torch.topk(
                    prob[i][:seq_len[i]],
                    k=torch.div(seq_len[i], 16, rounding_mode='floor') + 1,
                    largest=True)
                tmp = torch.mean(tmp).view(1)
            else:
                raise NotImplementedError()
            v_probs = torch.cat((v_probs, tmp))
        return v_probs

    @abstractmethod
    def forward(self, x, seq_len):
        """Defines the computation performed at every call."""

    def get_parameters(self, base_lr):
        return [{'params': self.parameters()}]
