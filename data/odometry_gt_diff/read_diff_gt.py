# -- coding: utf-8 --

import numpy as np

# @Time : 2022/10/15 15:30
# @Author : Zhiheng Feng
# @File : read_diff_gt.py
# @Software : PyCharm

"""
    本文件夹中的数据表示每个索引序列中相邻两帧位姿变换的真值,shape为[N,12]
"""


def read_diff(pose_path: str, index: int) -> np.ndarray:
    """
    注意返回的是index-th -> (index+1)-th相机位姿的变换,点云变换的话需要求对返回值求逆
    Args:
        pose_path: 路径
        index: 帧

    Returns: 从index-th帧相机到第(index+1)-th相机位姿的变换

    """
    poses = np.load(pose_path)
    pose = poses[index]
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    pose = np.concatenate([pose, filler], axis=-1)
    pose = pose.reshape((4, 4))
    return pose


if __name__ == '__main__':
    poses = np.load('00_diff.npy')
    print(poses.shape)
    pose0 = poses[0]
    print(pose0.reshape((3,4)))
