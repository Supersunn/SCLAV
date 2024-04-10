from fvcore.common.registry import Registry
from Viodetect.utils.build import build_from_cfg

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    CosineAnnealingWarmRestarts

OPTIMIZERS = Registry("optimizer")
LR_SCHEDULER = OPTIMIZERS

OPTIMIZERS.register(SGD)
OPTIMIZERS.register(Adam)
LR_SCHEDULER.register(ReduceLROnPlateau)
LR_SCHEDULER.register(MultiStepLR)
LR_SCHEDULER.register(CosineAnnealingWarmRestarts)

def build_optimizer(cfg, parameters):
    return build_from_cfg(cfg, OPTIMIZERS, parameters)


def build_lr_scheduler(cfg, optimizer):
    return build_from_cfg(cfg, LR_SCHEDULER, optimizer)
