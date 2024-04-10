import torch.nn as nn


def conv1d(in_channels: int, out_channels: int, kernel_size: int = 1):
    return nn.Conv1d(in_channels,
                     out_channels,
                     kernel_size=(kernel_size,),
                     stride=(1,),
                     padding=kernel_size // 2, bias=False)
