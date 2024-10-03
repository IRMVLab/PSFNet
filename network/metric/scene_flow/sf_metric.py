# -*- coding:UTF-8 -*-

import numpy as np


def evaluate_3d(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_3d_cls(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    if len(sf_pred.shape) > 2:
        sf_pred = np.squeeze(sf_pred, 0)
    if len(sf_gt.shape) > 2:
        sf_gt = np.squeeze(sf_gt, 0)

    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)  # (N,)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)  # (N,)
    relative_err = l2_norm / (sf_norm + 1e-4)  # (N,)

    cls_relax = np.logical_or(l2_norm < 0.1, relative_err < 0.1)
    cls = cls_relax.astype(np.int32)  # (N,)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = cls_relax.astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier, cls


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d