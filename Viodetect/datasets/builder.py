from fvcore.common.registry import Registry
from Viodetect.utils.build import build_from_cfg
from torch.utils.data import DataLoader
from itertools import cycle

DATASETS = Registry("dataset")
DATALOADERS = Registry("dataloader")

DATALOADERS.register(DataLoader)


@DATALOADERS.register()
def SemiDataLoader(data_loader_main, data_loader_sec):
    return zip(build_dataloader(data_loader_main),
               cycle(build_dataloader(data_loader_sec)))


def build_dataloader(cfg):
    if cfg['type'] == 'DataLoader':
        args = cfg.copy()
        args.pop('type')
        dataset = args.pop('dataset')
        dataset = build_dataset(dataset)
        return DataLoader(dataset, **args)
    else:
        return build_from_cfg(cfg, DATALOADERS)


def build_dataset(cfg):
    return build_from_cfg(cfg, DATASETS)
