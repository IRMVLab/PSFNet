# -*- coding:UTF-8 -*-

import numpy as np

from dataset.base_preprocess import BaseRangeLimit, BaseRandomSample


class RangeLimit(BaseRangeLimit):
    def __init__(self, *, x_range=(-30, 30), y_range=(-1, 1.4), z_range=(0, 35)):
        super().__init__(x_range=x_range, y_range=y_range, z_range=z_range)

    def __call__(self, data_dict):
        lidar1, lidar2 = list(map(self.limit_range, [data_dict['lidar1'], data_dict['lidar2']]))
        data_dict['lidar1'] = lidar1
        data_dict['lidar2'] = lidar2
        return data_dict


class RandomSample(BaseRandomSample):
    def __init__(self, *, num_points, allow_less_points):
        super().__init__(num_points, allow_less_points)

    def __call__(self, data_dict):
        lidar1, lidar2 = list(map(self.random_sample, [data_dict['lidar1'], data_dict['lidar2']]))
        data_dict['lidar1'] = lidar1
        data_dict['lidar2'] = lidar2
        return data_dict


class ShakeAug(object):
    def __init__(self, x_clip, y_clip, z_clip):
        super().__init__()
        self.x_clip = x_clip
        self.y_clip = y_clip
        self.z_clip = z_clip

    def __call__(self, data_dict):
        cur_aug_matrix = aug_matrix(self.x_clip, self.y_clip, self.z_clip)
        lidar1 = data_dict['lidar1']
        T_gt_lidar = data_dict['T_gt_lidar']
        lidar1 = np.concatenate([lidar1, np.ones((lidar1.shape[0], 1))], axis=-1)
        lidar1 = lidar1 @ cur_aug_matrix.T
        lidar1 = lidar1[:, :3]
        T_gt_lidar = T_gt_lidar @ np.linalg.inv(cur_aug_matrix)
        data_dict['lidar1'] = lidar1
        data_dict['T_gt_lidar'] = T_gt_lidar
        return data_dict


def aug_matrix(x_clip, y_clip, z_clip):
    anglex = np.clip(0.01 * np.random.randn(), -x_clip, x_clip).astype(np.float32) * np.pi / 4.0
    angley = np.clip(0.05 * np.random.randn(), -y_clip, y_clip).astype(np.float32) * np.pi / 4.0
    anglez = np.clip(0.01 * np.random.randn(), -z_clip, z_clip).astype(np.float32) * np.pi / 4.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])

    # 3*3
    R_trans = Rx.dot(Ry).dot(Rz)

    xx = np.clip(0.1 * np.random.randn(), -0.2, 0.2).astype(np.float32)
    yy = np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(np.float32)
    zz = np.clip(0.5 * np.random.randn(), -1, 1).astype(np.float32)

    add_3 = np.array([[xx], [yy], [zz]])
    T_trans = np.concatenate([R_trans, add_3], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  
    T_trans = np.concatenate([T_trans, filler], axis=0)

    return T_trans
