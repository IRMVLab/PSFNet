# -- coding: utf-8 --

import torch
import torch.nn as nn

from network.pointconv_util import index_points_group, square_distance


class SceneFlowLossModel(nn.Module):
    def __init__(self, weights = (0.02, 0.04, 0.08, 0.16), f_curvature=0.3, f_smoothness=1.0, f_chamfer=1.0):
        super().__init__()
        self.weights = weights
        self.f_curvature = f_curvature
        self.f_smoothness = f_smoothness
        self.f_chamfer = f_chamfer

    def forward(self, pc1, pc2, pred_flows):
        chamfer_loss = []
        smoothness_loss = []
        curvature_loss = []
        for cur_pc1, cur_pc2, cur_flow, weight in zip(pc1, pc2, pred_flows, self.weights):
            # compute curvature
            cur_pc1_warp = cur_pc1 + cur_flow
            cur_pc2_curvature = curvature(cur_pc2)

            moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)
            inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
            curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim=2).sum(dim=1).mean()

            # chamfer loss
            dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
            chamferLoss = dist1.sum(dim=1).mean() + dist2.sum(dim=1).mean()

            # smoothness loss
            smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim=1).mean()

            chamfer_loss.append(weight * chamferLoss)
            smoothness_loss.append(weight * smoothnessLoss)
            curvature_loss.append(weight * curvatureLoss)

        sf_loss = self.f_chamfer * sum(chamfer_loss) + self.f_smoothness * sum(
            smoothness_loss) + self.f_curvature * sum(curvature_loss)

        return sf_loss


def curvature(pc):
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim=-1, largest=False, sorted=False)  # B N 10 3
    grouped_pc = index_points_group(pc, kidx).cuda(device=pc.device)
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim=2) / 9.0
    return pc_curvature  # B N 3


def computeChamfer(pc1, pc2):
    sqrdist12 = square_distance(pc1, pc2)  # B N M
    # chamferDist
    dist1, _ = torch.topk(sqrdist12, 1, dim=-1, largest=False, sorted=False)
    dist2, _ = torch.topk(sqrdist12, 1, dim=1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)
    return dist1, dist2


def curvatureWarp(pc, warped_pc):
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim=-1, largest=False, sorted=False)  # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim=2) / 9.0
    return pc_curvature


def computeSmooth(pc1, pred_flow):
    # sf_mask  B,N
    sqrdist = square_distance(pc1, pc1)  # B N N

    # Smoothness
    _, kidx = torch.topk(sqrdist, 9, dim=-1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx)  # B N 9 3
    diff_flow = torch.norm(grouped_flow.contiguous() - pred_flow.unsqueeze(2).contiguous(), dim=3).sum(dim=2) / 8.0

    return diff_flow


def interpolateCurvature(pc1, pc2, pc2_curvature):
    B, N, _= pc1.shape
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2)  # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim=-1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx)  # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim=2, keepdim=True)
    weight = (1.0 / (dist + 1e-8)) / norm

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim=2)
    return inter_pc2_curvature
