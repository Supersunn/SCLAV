import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.spatial.distance import pdist, squareform


def make_dist_mat(batch_size, max_seq_len, device):
    # arith = np.arange(max_seq_len).reshape(-1, 1)
    # dist = pdist(arith, metric='cityblock').astype(np.float32)
    # dist = torch.from_numpy(squareform(dist)).to(device)
    # dist = torch.exp(-dist / torch.exp(torch.tensor(1.)))
    # dist = torch.unsqueeze(dist, 0).repeat(batch_size, 1, 1).to(device)
    # return dist
    pass


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, dropout: float = 0.5, max_len=2500):
        # max_len bigger than the number of segments
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # build a big enough P
        P = np.zeros((1, max_len, num_hidden))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1, 1) / np.power(10000, np.arange(0, num_hidden, 2,
                                               dtype=np.float32) / num_hidden)
        P[:, :, 0::2] = np.sin(X)
        P[:, :, 1::2] = np.cos(X)

        self.P = torch.tensor(P, dtype=torch.float32)

    def forward(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X, **kwargs)


class CoAttention(nn.Module):
    def __init__(self, in_channels: int, pos_emb: bool = False,
                 skip_connection: str = 'cross', use_layer_norm: bool = True,
                 use_sqrt: bool = True):
        super().__init__()

        self.w_k = nn.Linear(in_channels, in_channels, True)
        self.w_q = nn.Linear(in_channels, in_channels, True)

        self.w_v1 = nn.Linear(in_channels, in_channels, True)
        self.w_v2 = nn.Linear(in_channels, in_channels, True)

        self.channels = in_channels

        self.layer_norm = nn.LayerNorm(in_channels)

        self.use_layer_norm = use_layer_norm

        self.use_sqrt = use_sqrt

        self.pos_emb = pos_emb

        self.skip_connection = skip_connection

        if pos_emb:
            self.pos_encoding = PositionalEncoding(in_channels)

    def forward(self, x1, x2):
        x1 = x1.transpose(1, 2)     # v     # input:B*C*T   output:B*C*T
        x2 = x2.transpose(1, 2)     # a

        if self.pos_emb:
            x1 = self.pos_encoding(x1)
            x2 = self.pos_encoding(x2)

        q = self.w_q(x1)
        k = self.w_k(x2)

        # v1 = self.w_v1(q)          
        # v2 = self.w_v2(k)
        v1 = self.w_v1(x1)          
        v2 = self.w_v2(x2)

        attn = torch.bmm(q, k.transpose(1, 2))

        if self.use_sqrt:
            attn = attn / torch.sqrt(torch.tensor(self.channels)).to(q.device)

        attn_k = torch.softmax(attn, dim=-1)                # row
        attn_q = torch.softmax(attn, dim=1).transpose(1, 2) # col

        vk = torch.bmm(attn_k, v2)
        vq = torch.bmm(attn_q, v1)

        if self.skip_connection == 'cross':
            if self.use_layer_norm:
                result1 = self.layer_norm(vk + x1)
                result2 = self.layer_norm(vq + x2)
            else:
                result1 = vk + x1
                result2 = vq + x2
        elif self.skip_connection == 'straight':
            if self.use_layer_norm:
                result1 = self.layer_norm(vk + x2)
                result2 = self.layer_norm(vq + x1)
            else:
                result1 = vk + x2
                result2 = vq + x1
        else:
            if self.use_layer_norm:
                result1 = self.layer_norm(vk)
                result2 = self.layer_norm(vq)
            else:
                result1 = vk
                result2 = vq

        return result1.transpose(1, 2), result2.transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, pos_emb: bool = False,
                 use_layer_norm: bool = True, use_sqrt: bool = True):
        super().__init__()

        self.w_k = nn.Linear(in_channels, in_channels, bias=False)
        self.w_q = nn.Linear(in_channels, in_channels, bias=False)

        self.w_v = nn.Linear(in_channels, in_channels, bias=False)

        self.channels = in_channels

        self.layer_norm = nn.LayerNorm(in_channels)

        self.use_layer_norm = use_layer_norm

        self.use_sqrt = use_sqrt

        self.pos_emb = pos_emb
        if pos_emb:
            self.pos_encoding = PositionalEncoding(in_channels)

    def forward(self, x):
        # Input: BxCxT
        x = x.transpose(1, 2)  # BxTxC

        if self.pos_emb:
            x = self.pos_encoding(x)

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        attn = torch.bmm(q, k.transpose(1, 2))

        if self.use_sqrt:
            attn = attn / torch.sqrt(torch.tensor(self.channels)).to(v.device)

        if self.pos_emb:
            pos = make_dist_mat(x.shape[0], x.shape[1], x.device)
            attn_ = (F.softmax(attn, dim=-1) + pos) / 2
        else:
            attn_ = F.softmax(attn, dim=-1)

        result = torch.bmm(attn_, v) + x
        if self.use_layer_norm:
            return self.layer_norm(result).transpose(1, 2)  # BxCxT
        else:
            return result.transpose(1, 2)
