from torch import nn as nn

from Viodetect.models.builder import build_neck, build_head, build_loss, MODELS
from .base import BaseTask
import torch


@MODELS.register()
class VAWeakDetectionFromFeature(BaseTask):
    def __init__(self, fuse, head, loss) -> None:
        super().__init__()
        self.fuse = build_neck(fuse)
        self.head = build_head(head)
        self.loss = build_loss(loss)

    def forward(self, batch):
        label = batch.get('labels', None)
        assert label is not None
        seq_len = batch.get('seq_len', None)
        assert seq_len is not None
        outputs = self.inference(batch)
        loss = self.loss(outputs, label, seq_len)
        return loss, outputs

    def inference(self, batch):
        video = batch.get('imgs', None)
        audio = batch.get('audios', None)
        seq_len = batch.get('seq_len', None)
        assert video is not None or audio is not None
        feature = self.fuse(video, audio)

        if type(feature) is dict:
            for k in feature:
                feature[k] = feature[k][:, :torch.max(seq_len), :]
        else:
            feature = feature[:, :torch.max(seq_len), :]
        return self.head(feature, seq_len)

    def get_parameters(self, base_lr):
        return self.fuse.get_parameters(base_lr) + self.head.get_parameters(
            base_lr)
