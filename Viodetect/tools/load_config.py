import importlib
import os


def load_cfg(cfg_name: str, custom_name: str = ""):
    cfg = importlib.import_module(cfg_name)
    if custom_name:
        cfg.name = custom_name
    else:
        cfg.name = cfg_name

    cfg.save_dir = os.path.join(cfg.work_dir, 'checkpoints', cfg.name)
    cfg.log_dir = os.path.join(cfg.work_dir,'logs',cfg.name)
    return cfg
