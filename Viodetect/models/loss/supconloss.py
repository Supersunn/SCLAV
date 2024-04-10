import torch
import torch.nn as nn
import torch.nn.functional as F

from .topk import TopKLoss
from ..builder import LOSSES
from ..components.losses import get_sup_con_loss,get_seq_mask,get_cross_con_loss,get_flatten_label

@LOSSES.register()
class SupConLoss(TopKLoss):

    def __init__(self, gamma: float = 0.05,gamma2:float=1.0,threshold:float=0.9, temperature=0.5, scale_by_temperature=True,**kwargs):
        super(SupConLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.gamma2 = gamma2
        self.threshold = threshold
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def get_mutual_loss(self, prob, prob2, seq_len,labels):
        input_shape = prob.shape
        device = prob.device
        seq_mask = get_seq_mask(input_shape, seq_len, device)

        prob = prob.flatten()
        prob2 = prob2.flatten()

        threshold = self.threshold
        # ul_loss = get_sup_con_loss(prob, prob2, seq_mask, device,
        #                             threshold=threshold,
        #                             temperature=self.temperature,scale_by_temperature=self.scale_by_temperature)
        # ul_loss += get_sup_con_loss(prob2, prob, seq_mask, device,
        #                             threshold=threshold,
        #                             temperature=self.temperature,scale_by_temperature=self.scale_by_temperature)
        
        
        mask = get_flatten_label(input_shape,labels)
        ul_loss = get_cross_con_loss(prob, prob2,device,threshold=threshold,temperature=self.temperature,scale_by_temperature=self.scale_by_temperature,labels=mask)
        # co-branch、v-branch

        return ul_loss

    def forward(self, outputs, labels, seq_len):
        prob = outputs['prob'].flatten(1)  # BxT
        prob2 = outputs['prob2'].flatten(1)  # v-branch

        loss = self.calculate_mil_loss(prob, labels, seq_len)   # Top-k
        loss += self.gamma2 * self.calculate_mil_loss(prob2, labels, seq_len)

        loss += self.gamma * self.get_mutual_loss(prob, prob2, seq_len,labels=labels)

        return loss