import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        loss_type = kwargs.get('mil_loss', 'bce')
        if loss_type == 'bce':
            self.loss = nn.BCELoss()
        elif loss_type == 'mse':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError()
