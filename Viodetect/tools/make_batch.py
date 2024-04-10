import torch


def make_batch(device,n: int = 2, t: int = 16):

    imgs = torch.rand((n, t, 1024))
    audios = torch.rand((n, t, 128))
    seq_len = torch.randint(t // 2, t, (n, ))
    labels = torch.randint(0, 2, (n, )).float()

    return dict(imgs=imgs.to(device), audios=audios.to(device), seq_len=seq_len.to(device),
                labels=labels.to(device))
