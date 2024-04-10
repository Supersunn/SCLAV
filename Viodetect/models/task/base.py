from torch import nn as nn
from abc import ABCMeta, abstractmethod


class BaseTask(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, batch):
        ...

    @abstractmethod
    def inference(self, batch):
        ...

    @abstractmethod
    def get_parameters(self, base_lr):
        ...