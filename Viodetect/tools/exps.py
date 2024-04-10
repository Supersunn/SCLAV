from run_net import train, test
from load_config import load_cfg
# import time
# import numpy as np
# import os
# import datetime


def exp_train():
    base_name = 'configs.'
    name = 'colstm_supcon'
    cfg = load_cfg(base_name + name, 'exp_' + name)
    cfg.model['loss']['gamma'] = 1.5          #0.55
    cfg.model['loss']['gamma2'] = 1.4         #1.00
    # cfg.model['loss']['threshold'] = 0.5
    cfg.log_dir = f"./workdirs/logs/{name}0409"
    cfg.save_dir = f"./workdirs/checkpoints/{name}0409"
    print(train(cfg, logger=False))

def exp_test():
    model_path = '/home/sunchao/SCLAV/workdirs/checkpoints/exp0323_0.8493_gamma=1.190_gamma2=0.556_tau=0.219/epoch=13-step=2169.ckpt'
    exp_name = 'colstm_supcon'
    full_name = 'configs.' + exp_name

    window_size = [1, 3, 5, 7, 9, 11]
    cfg = load_cfg(full_name, f'window_size')

    result = []
    for w in window_size:
        if w == 1:
            result.append(test(cfg, model_path, smooth=False))
        else:
            result.append(test(cfg, model_path, smooth=True, window=w))
    print(result)


if __name__ == '__main__':

    exp_train()      # train
    # exp_test()      # test