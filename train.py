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
from utils.AverageMeter import AverageMeter
from utils.logging_utils import get_logger
from utils.config_utils import parse_args
from network.loss.odometry_loss import OdometryLossModel
from network.loss.scene_flow_loss import SceneFlowLossModel


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

    logger = get_logger('psfnet', log_file=os.path.join(output_path, 'train.log'))
    logger.info(cfg)
    set_random_seed(seed=cfg.SEED, use_cuda='cuda' in cfg.train_options.devices[0] and torch.cuda.is_available(), deterministic=True)
    psf_net = PSFNet(cfg.model_option, cfg.train_option, logger)

    pose_loss_func = OdometryLossModel()
    pose_loss_func.to(psf_net.devices[0])

    sf_loss_func = SceneFlowLossModel()
    sf_loss_func.to(psf_net.devices[0])

    kod_train_loader = build_dataloader(cfg.dataset_option.train.kod)
    all_step = len(kod_train_loader)
    logger.info(f"train dataset has {kod_train_loader.dataset.__len__()} samples, {all_step} in dataloader")

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

    pose_eval_seq = cfg.dataset_option.pose_eval.seqs_list
    pose_eval_dataset_dict = dict()
    for seq in pose_eval_seq:
        key_name = f'{seq:02d}'
        dataset_option = copy.deepcopy(cfg.dataset_option.pose_eval.kod)
        dataset_option.dataset.update({'seqs_list': [seq]})
        data_loader = build_dataloader(dataset_option)
        pose_eval_dataset_dict[key_name] = dict()
        pose_eval_dataset_dict[key_name]['data_loader'] = data_loader
        logger.info(f'seq {key_name} eval dataset has {len(data_loader.dataset)} samples,{len(data_loader)} in dataloader')
        key_list = ['rotate', 'trans']
        pose_eval_dataset_dict[key_name]['key_list'] = key_list
        metrics = pd.DataFrame(columns=key_list, dtype=float)
        metrics.index.name = 'epoch'
        pose_eval_dataset_dict[key_name]['metrics'] = metrics

    global_state = psf_net.global_state
    start_epoch = global_state.get('epoch', 0)
    cur_status = global_state.get('cur_status', 'all')
    psf_net.cur_status = cur_status

    logger.info(f"start_epoch is {start_epoch}, start status is {cur_status}")

    pose_pred_root = os.path.join(output_path, 'pose_pred')
    pose_visual_root = os.path.join(output_path, 'pose_visual')
    pose_xlsx_root = os.path.join(output_path, 'pose_xlsx')
    sf_xlsx_root = os.path.join(output_path, 'sf_xlsx')


    best_epoch = start_epoch
    eta_meter = AverageMeter()
    try:
        for epoch in range(start_epoch + 1, psf_net.max_epoch):
            psf_net.train(kod_train_loader, sf_loss_func, pose_loss_func, eta_meter, epoch)

            if epoch % cfg.train_option.val_interval == 0 or epoch in [end - 1 for _, end in cfg.train_option.train_schedule]:
                global_state['epoch'] = epoch
                global_state['cur_status'] = psf_net.cur_status

                pose_metric, h_mean = psf_net.eval_pose(pose_eval_dataset_dict, epoch, pose_pred_root, pose_visual_root, pose_xlsx_root)
                global_state['pose_metric'] = pose_metric
                if psf_net.cur_status != 'pose':
                    sf_metric = psf_net.eval_sf(sf_eval_dataset_dict, epoch, sf_xlsx_root)
                    global_state['sf_metric'] = sf_metric

                if cfg.train_options.ckpt_save_type == 'HighestAcc' and psf_net.cur_status == 'pose':
                    if h_mean < global_state.get('pose_hmean', 1000):
                        best_epoch = epoch
                        global_state['pose_hmean'] = h_mean
                        net_save_path = f"{ckpt_root}/best.pth.tar"
                else:
                    net_save_path = f"{ckpt_root}/{epoch:04d}.pth.tar"
                psf_net.save_checkpoint(net_save_path, global_state=global_state)

                if cfg.train_options.ckpt_save_type == 'HighestAcc':
                    logger.info(f"best epoch is {best_epoch}, pose_hmean is {global_state.get('pose_hmean', 1000)}")

    except KeyboardInterrupt:
        net_save_path = f"{output_path}/final.pth.tar"
        psf_net.save_checkpoint(net_save_path, global_state=global_state)
    except:
        error_msg = traceback.format_exc()
        logger.error(error_msg)
    finally:
        pass

if __name__ == '__main__':
    main()
