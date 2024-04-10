from torch.utils.data import Dataset
import json
import numpy as np
import torch

from .utils import process_feat
from .builder import DATASETS


@DATASETS.register()
class VAFeatureDataset(Dataset):
    def __init__(self,
                 annotation_file: str,
                 max_seq_len: int,
                 train_mode: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.annotations = annotation_file
        self.label_map = {'normal': 0, 'abnormal': 1}
        self.max_seq_len = max_seq_len
        self.train_mode = train_mode
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        with open(self.annotations, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data

    def __getitem__(self, index):
        item = self.data_infos[index]
        imgs = np.load(item['video'])
        audios = np.load(item['audio'])
        label = self.label_map[item['label']]

        if self.train_mode:
            d_imgs = imgs.shape[1]
            feat = np.concatenate((imgs, audios), axis=1)
            feat, seq_len = process_feat(feat,
                                         self.max_seq_len,
                                         is_random=False)
            imgs = feat[:, :d_imgs]
            audios = feat[:, d_imgs:]
            return {
                'imgs': torch.from_numpy(imgs),
                'audios': torch.from_numpy(audios),
                'labels': torch.tensor(label, dtype=torch.float32),
                'seq_len': torch.tensor(seq_len, dtype=torch.long)
            }
        else:
            data = {
                'imgs': torch.from_numpy(imgs),
                'audios': torch.from_numpy(audios),
                'labels': torch.tensor(label, dtype=torch.float32),
                'seq_len': torch.tensor(imgs.shape[0], dtype=torch.long),
                'index': index // 5
            }
            return data

    def __len__(self):
        return len(self.data_infos)
