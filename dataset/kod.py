# -- coding: utf-8 --
import os
import bisect
import yaml
import numpy as np
import torch.utils.data as data

from utils.json_process import json2dict
from utils.euler_tools import euler2quat, mat2euler
from dataset.kod_preprocess import RangeLimit, RandomSample, ShakeAug


class KOD(data.Dataset):

    def __init__(self, dataset_path, seqs_list, pre_processes, check_seq_len=True,
                 gt_root='data/odometry_gt_diff', vel2cam_root='data/vel_to_cam_Tr.json'):
        super().__init__()

        self.dataset_root = dataset_path
        
        seqs_list = sorted(seqs_list)
        self.seqs_list = seqs_list
        self.gt_root = gt_root

        kITTI_seqs_ = [4540, 1100, 4660, 800, 270, 2760, 1100, 1100, 4070, 1590, 1200]
        expected_seqs_len = np.array([kITTI_seqs_[seq_index] + 1 for seq_index in seqs_list])
        seqs_length = np.array(self.get_seqs_len())


        if check_seq_len:
            wrong_index = [i for i in range(len(seqs_length)) if not expected_seqs_len[i] == seqs_length[i]]
            wrong_seq = [seqs_list[index] for index in wrong_index]
            wrong_seq_len = [seqs_length[index] for index in wrong_index]
            expect_wrong_seq_len = [expected_seqs_len[index] for index in wrong_index]
            if wrong_index:
                raise ValueError(
                    f'the num of sequences {wrong_seq} is {wrong_seq_len}, but expected {expect_wrong_seq_len}')

        self.seqs_len_cumsum = [0] + list(np.cumsum(seqs_length))

   
        vel2cam_dict = json2dict(vel2cam_root)
        self.vel_to_cam_Tr = [np.array(vel2cam_dict[f'Tr_{seq_index:02d}']) for seq_index in seqs_list]


        self._init_pre_processes(pre_processes)
        
        self.gt_Tr = [self.read_diff_gt(os.path.join(self.gt_root, f'{seq_index:02d}_diff.npy')) for seq_index in self.seqs_list]
            


    def __len__(self):
        return self.seqs_len_cumsum[-1]


    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):

        if index in self.seqs_len_cumsum:
            seq_index = self.seqs_len_cumsum.index(index)
            fn2 = 0
            fn1 = 0
        else:
            seq_index = bisect.bisect(self.seqs_len_cumsum, index) - 1
            fn2 = index - self.seqs_len_cumsum[seq_index]
            fn1 = fn2 - 1


        data_dict = dict()

        lidar_path = os.path.join(self.dataset_root, f'{self.seqs_list[seq_index]:02d}', 'velodyne')
        fn1_dir = os.path.join(lidar_path, f'{fn1:06d}.bin')
        fn2_dir = os.path.join(lidar_path, f'{fn2:06d}.bin')
        lidar1 = self.read_lidar(fn1_dir)
        lidar2 = self.read_lidar(fn2_dir)
        cam2_to_cam1 = self.gt_Tr[seq_index][fn2]
        cam1_to_cam2 = np.linalg.inv(cam2_to_cam1)   # p2 = T12 * p1
       
        Tr = self.vel_to_cam_Tr[seq_index]
        lidar1 = self.vel2cam(lidar1, Tr)
        lidar2 = self.vel2cam(lidar2, Tr)
        
        data_dict['lidar1'] = lidar1
        data_dict['lidar2'] = lidar2
        data_dict['T_gt_lidar'] = cam1_to_cam2

        data_dict = self.apply_pre_processes(data_dict)

        T_gt_lidar = data_dict['T_gt_lidar']
        R_gt = T_gt_lidar[:3, :3]
        t_gt = T_gt_lidar[:3, 3]
        z_gt, y_gt, x_gt = mat2euler(M=R_gt)
        q_gt = euler2quat(z=z_gt, y=y_gt, x=x_gt)
        
        t_gt = t_gt.astype(np.float32)  # (3,)
        q_gt = q_gt.astype(np.float32)  # (4,)

        lidar1 = data_dict['lidar1'].astype(np.float32)
        lidar2 = data_dict['lidar2'].astype(np.float32)
        lidar1_norm = lidar1
        lidar2_norm = lidar2

        return {'lidar1': lidar1, 'lidar2': lidar2, 'norm1': lidar1_norm, 'norm2': lidar2_norm,
                'q_gt': q_gt, 't_gt': t_gt}

    def get_seqs_len(self):
        results = []
        for seq_index in self.seqs_list:
            lidar_path = os.path.join(self.dataset_root, f'{seq_index:02d}', 'velodyne')
            seq_len = len(os.listdir(lidar_path))
            results.append(seq_len)
        return results

    def read_diff_gt(self, pose_path: str) -> np.ndarray:
        poses = np.load(pose_path)   # (N, 12)
        poses = poses.reshape(-1, 3, 4)
        paddings = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 1, 4)
        paddings = np.repeat(paddings, poses.shape[0], axis=0)
        poses = np.concatenate([poses, paddings], axis=1)  # (N, 4, 4)
        return poses

    def read_lidar(self, lidar_path: str):
        lidar = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        lidar = lidar[:, :3]
        return lidar


    def vel2cam(self, lidar, Tr):
        lidar =  np.concatenate([lidar, np.ones((lidar.shape[0], 1))], axis=-1)
        lidar = lidar @ Tr.T
        lidar = lidar[:,:3]
        return lidar


