# -*- coding:UTF-8 -*-

import os
import time
import torch
import logging
import datetime
import numpy as np
import pandas as pd

from addict import Dict
from tqdm import tqdm

from network.model import build_model
from utils.base_net import BaseNet
from utils.euler_tools import quat2mat
from utils.AverageMeter import AverageMeter
from utils.metric_analysis import odo_metric_analysis
from network.metric.scene_flow.geometry import get_batch_2d_flow
from network.metric.scene_flow.sf_metric import evaluate_2d, evaluate_3d
from network.metric.odometry.odometry_metric import kittiOdomEval


def dict2list(input_dict: dict, key_order: tuple) -> list:
    return [input_dict[k] for k in key_order]


class PSFNet(BaseNet):

    def __init__(self, model_config: Dict, train_config: Dict, logger: logging.Logger):
        psf_model = build_model(model_config)
        super().__init__(psf_model, train_config, logger)
        self.train_config = train_config
        self.lr_clip = train_config.learning_rate_clip
        self.print_interval = train_config.print_interval
        self.max_epoch = train_config.epochs
        self.train_schedule = train_config.train_schedule
        self.all_status = ['pose', 'flow', 'all']
        self.cur_status = 'all'

    def train(self, train_loader, sf_loss_func, pose_loss_func, eta_meter, epoch):
 
        self.net = self.net.train()
        start = time.time()

        pose_loss_record = 0
        sf_loss_record = 0

        all_step = len(train_loader)
        global_step = len(train_loader) * epoch
        if epoch == self.train_schedule[1][0]:
            self.optimizer = self.creat_optimizer(self.train_config.optimizer)
            self.scheduler = self.creat_scheduler(self.train_config.scheduler)
        else:
            lr = max(self.optimizer.param_groups[0]['lr'], self.lr_clip)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        now_step_index = 0
        for j, (start_epoch, end_epoch) in enumerate(self.train_schedule):
            if start_epoch <= epoch < end_epoch:
                now_step_index = j
                break
        self.cur_status = self.all_status[now_step_index]
        for i, batch_data in enumerate(train_loader):
            for key, value in batch_data.items():
                if value is not None and isinstance(value, torch.Tensor):
                    batch_data[key] = value.to(self.devices[0])

            pre_q_list, pre_t_list, _, pc1_list, pc2_list, pre_flow_list, *_ = self.net(batch_data['lidar1'],
                                                                                        batch_data['lidar2'],
                                                                                        batch_data['norm1'],
                                                                                        batch_data['norm2'],
                                                                                        state=self.cur_status)
            
            cur_pose_loss = 0
            cur_sf_loss = 0

            self.optimizer.zero_grad()
            if self.cur_status == 'pose':
                pose_loss_func.train()
                pose_loss = pose_loss_func(pre_q_list, pre_t_list, batch_data['q_gt'], batch_data['t_gt'], self.net.w_x, self.net.w_q)
                loss = pose_loss
                cur_pose_loss = pose_loss.item()
            elif self.cur_status == 'flow':
                sf_loss_func.train()
                sf_loss = sf_loss_func(pc1_list, pc2_list, pre_flow_list)
                loss = sf_loss
                cur_sf_loss = sf_loss.item()
            else:
                pose_loss_func.train()
                sf_loss_func.train()
                pose_loss = pose_loss_func(pre_q_list, pre_t_list, batch_data['q_gt'], batch_data['t_gt'], self.net.w_x, self.net.w_q)
                sf_loss = sf_loss_func(pc1_list, pc2_list, pre_flow_list)
                loss = pose_loss + sf_loss
                cur_pose_loss = pose_loss.item()
                cur_sf_loss = sf_loss.item()

            pose_loss_record += cur_pose_loss
            sf_loss_record += cur_sf_loss

            loss.requires_grad_(True)
            loss.backward()
            self.optimizer.step()

            train_batch_time = float(time.time() - start)
            eta_meter.update(train_batch_time)

            if (i + 1) % self.print_interval == 0 or i == all_step - 1:
                eta_sec = ((self.max_epoch - epoch) * all_step - i - 1) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                self.logger.info(f"Epoch[{epoch}/{self.max_epoch}] - "
                                 f"step[{i + 1}/{all_step}] - "
                                 f"status:{self.cur_status} - "
                                 f"lr:{self.get_learing_rate()} - "
                                 f"odometry loss:{cur_pose_loss:.4f} - "
                                 f"sf loss:{cur_sf_loss:.4f} - "
                                 f"time:{train_batch_time:.4f} - "
                                 f"eta: {eta_sec_format}")
                start = time.time()
            global_step += 1

        self.logger.info(f"Mean odometry train_loss: {pose_loss_record / all_step}")
        self.logger.info(f"Mean scene flow train_loss: {sf_loss_record / all_step}")
        self.scheduler.step()

    def eval_pose(self, eval_data_dict, cur_epoch, pred_save_root, visual_save_root, xlsx_save_root):
        epoch = cur_epoch
        os.makedirs(pred_save_root, exist_ok=True)
        os.makedirs(visual_save_root, exist_ok=True)
        os.makedirs(xlsx_save_root, exist_ok=True)
        cur_status = self.cur_status
        if cur_status == 'flow':
            cur_status = 'pose'
        for seq, value in eval_data_dict.items():
            self.logger.info(f" start to eval odometry dataset: {seq}")
            pred_save_path = os.path.join(pred_save_root, f'{seq}_pred.npy')
            self.eval_pose_one_seq(value['data_loader'], pred_save_path, status=cur_status)

        metric_cfg = Dict()
        metric_cfg.gt_dir = 'data/odometry_gt'
        metric_cfg.pre_result_dir = pred_save_root
        metric_cfg.visual_save_root = visual_save_root
        metric_cfg.eva_seqs = [key for key in eval_data_dict.keys()]
        metric_cfg.epoch = cur_epoch
        odo_metric = kittiOdomEval(metric_cfg)
        ave_metric = odo_metric.eval(toCameraCoord=False)

        metrics_list = []
        seq_list = []
        for seq, value in eval_data_dict.items():
            result_dict = ave_metric[seq]
            value['metrics'].loc[str(epoch)] = dict2list(result_dict, value['key_list'])
            metrics_list.append(value['metrics'])
            seq_list.append(seq)
            if xlsx_save_root:
                value['metrics'].to_excel(os.path.join(xlsx_save_root, f'{seq}.xlsx'))
        all_seqs_metrics = pd.concat(metrics_list, axis=1)
        all_seqs_metrics.columns = [f'{seq}_{metric_name}' for metric_name in eval_data_dict[seq]['key_list'] for seq in seq_list]
        odo_metric_analysis(all_seqs_metrics, eval_data_dict[list(eval_data_dict.keys())[0]]['key_list'])
        all_seqs_metrics.to_excel(os.path.join(xlsx_save_root, 'all_seqs.xlsx'))

        h_mean = 0
        for value in ave_metric.values():
            h_mean += (2*value['rotate']*value['trans']) / (value['rotate'] + value['trans'])
        h_mean /= len(ave_metric)
        
        return ave_metric, h_mean

    def eval_sf(self, eval_data_dict, cur_epoch, xlsx_save_root) -> dict:
        epoch = cur_epoch
        result = dict()
        cur_status = self.cur_status
        if cur_status == 'pose':
            cur_status = 'flow'
        for name, value in eval_data_dict.items():
            self.logger.info(f" start to eval sf dataset: {name}")
            result_dict = self.eval_sf_one_loader(value['data_loader'], eval_2d_flag=value['eval_2d'], status=cur_status)
            result[name] = result_dict
            value['metrics'].loc[str(epoch)] = dict2list(result_dict, value['key_list'])
            if xlsx_save_root:
                if not os.path.exists(xlsx_save_root):
                    os.makedirs(xlsx_save_root)
                value['metrics'].to_excel(os.path.join(xlsx_save_root, f'{name}.xlsx'))
        return result

    def eval_pose_one_seq(self, val_loader, save_pred_path, status):
        self.net = self.net.eval()
        with torch.no_grad():
            line = 0
            for batch_data in tqdm(val_loader):
                for key, value in batch_data.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.devices[0])
                pre_q_list, pre_t_list, *_ = self.net(batch_data['lidar1'],
                                                      batch_data['lidar2'],
                                                      batch_data['norm1'],
                                                      batch_data['norm2'],
                                                      state=status)

                pred_q = pre_q_list[0].cpu().numpy()
                pred_t = pre_t_list[0].cpu().numpy()

                for qq, tt in zip(pred_q, pred_t):
                    qq = qq.reshape(4)
                    tt = tt.reshape(3, 1)
                    RR = quat2mat(qq)
                    TT = np.concatenate([np.concatenate([RR, tt], axis=-1), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
                    TT = np.linalg.inv(TT)
                    if line == 0:
                        T_final = TT  # 4 4
                        T = T_final[:3, :]  # 3 4
                        T = T.reshape(1, 12)
                        line += 1
                    else:
                        T_final = np.matmul(T_final, TT)
                        T_current = T_final[:3, :]
                        T_current = T_current.reshape(1, 12)
                        T = np.append(T, T_current, axis=0)
            T = T.reshape(-1, 12)
            np.save(save_pred_path, T)

    def eval_sf_one_loader(self, val_loader, eval_2d_flag=False, status='all') -> dict:
        self.net = self.net.eval()
        epe3ds = AverageMeter()
        acc3d_stricts = AverageMeter()
        acc3d_relaxs = AverageMeter()
        outliers = AverageMeter()
        if eval_2d_flag:
            epe2ds = AverageMeter()
            acc2ds = AverageMeter()

        with torch.no_grad():
            for batch_data in tqdm(val_loader):
                for key, value in batch_data.items():
                    if value is not None and isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.devices[0])
                
                *_, pc1, pc2, pred_flows, _, _, label = self.net(batch_data['pc1'],
                                                                 batch_data['pc2'],
                                                                 batch_data['norm1'],
                                                                 batch_data['norm2'],
                                                                 label=batch_data['flow'],
                                                                 state=status)

                full_flow = pred_flows[0]
                sf_gt = label[0]
                pc1_ = pc1[0]

                pc1_np = pc1_.cpu().numpy()
                sf_np = sf_gt.cpu().numpy()
                pred_sf = full_flow.cpu().numpy()
                epe3d, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)
                epe3ds.update(epe3d)
                acc3d_stricts.update(acc3d_strict)
                acc3d_relaxs.update(acc3d_relax)
                outliers.update(outlier)

                if eval_2d_flag:
                    flow_pred, flow_gt = get_batch_2d_flow(pc1_np, pc1_np + sf_np, pc1_np + pred_sf, batch_data['path'])
                    epe2d, acc2d = evaluate_2d(flow_pred, flow_gt)
                    epe2ds.update(epe2d)
                    acc2ds.update(acc2d)
     
        if eval_2d_flag:
            return {
                'EPE3D': epe3ds.avg,
                'ACC3DS': acc3d_stricts.avg,
                'ACC3DR': acc3d_relaxs.avg,
                'Outliers3D': outliers.avg,
                'EPE2D': epe2ds.avg,
                'ACC2D': acc2ds.avg
            }
        else:
            return {'EPE3D': epe3ds.avg, 'ACC3DS': acc3d_stricts.avg, 'ACC3DR': acc3d_relaxs.avg, 'Outliers3D': outliers.avg}
