# -*- coding:UTF-8 -*-

import os
import copy
import torch
import random
import traceback
import datetime

import numpy as np
import pandas as pd
import pytz

from network.psf_net import PSFNet
from dataset import build_dataloader
from utils.logging_utils import get_logger
from utils.config_utils import parse_args

def set_random_seed(seed=3407, use_cuda=True, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def main():
    cfg = parse_args()

    output_root = cfg.train_option.output_save_dir
    tz = pytz.timezone('Asia/Shanghai')
    output_path = os.path.join(output_root, f'{datetime.datetime.now(tz).strftime("%m-%d-%H-%M")}')
    os.makedirs(output_path, exist_ok=True)
    ckpt_root = os.path.join(output_path, 'checkpoint')
    os.makedirs(ckpt_root, exist_ok=True)

    logger = get_logger('psfnet', log_file=os.path.join(output_path, 'eval.log'))
    logger.info(cfg)
    set_random_seed(seed=cfg.SEED, use_cuda='cuda' in cfg.train_options.devices[0] and torch.cuda.is_available(), deterministic=True)
    psf_net = PSFNet(cfg.model_option, cfg.train_option, logger)

    sf_eval_dataset_name = list(cfg.dataset_option.sf_eval.keys())
    sf_eval_dataset_dict = dict()
    for key_name in sf_eval_dataset_name:
        data_loader = build_dataloader(cfg.dataset_option.sf_eval[key_name])
        sf_eval_dataset_dict[key_name] = dict()
        sf_eval_dataset_dict[key_name]['data_loader'] = data_loader
        sf_eval_dataset_dict[key_name]['eval_2d'] = cfg.dataset_option.sf_eval[key_name]['eval_2d']
        logger.info(f'{key_name} eval dataset has {data_loader.dataset.__len__()} samples,{len(data_loader)} in dataloader')
        if sf_eval_dataset_dict[key_name]['eval_2d']:
            key_list = ('EPE3D', 'ACC3DS', 'ACC3DR', 'Outliers3D', 'EPE2D', 'ACC2D')
        else:
            key_list = ('EPE3D', 'ACC3DS', 'ACC3DR', 'Outliers3D')
        sf_eval_dataset_dict[key_name]['key_list'] = key_list
        metrics = pd.DataFrame(columns=key_list, dtype=float)
        metrics.index.name = 'epoch'
        sf_eval_dataset_dict[key_name]['metrics'] = metrics

    global_state = psf_net.global_state
    cur_status = global_state.get('cur_status', 'all')
    psf_net.cur_status = cur_status
    sf_xlsx_root = os.path.join(output_path, 'sf_xlsx')
    psf_net.eval_sf(sf_eval_dataset_dict, -1, sf_xlsx_root)

if __name__ == '__main__':
    main()
