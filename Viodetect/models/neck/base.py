from torch import nn as nn
from abc import ABCMeta, abstractmethod
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)


class BaseNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, video=None, audio=None):
        ...

    def get_parameters(self, base_lr):
        return [{'params': self.parameters()}]

    def init_weights(self):
        self.apply(weight_init)