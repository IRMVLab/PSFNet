# -- coding: utf-8 --
import torch
import torch.nn as nn

class OdometryLossModel(nn.Module):

    def __init__(self, weights: tuple = (0.2, 0.4, 0.8, 1.6)):
        super().__init__()
        self.weights = weights  # from fine to coarse


    def forward(self, pre_q_list, pre_t_list, q_gt, t_gt, w_x, w_q):
        loss_list = []

        for pre_q_norm, pre_t, weight in zip(pre_q_list, pre_t_list,
                                             self.weights):
            loss_q = torch.mean(
                torch.sqrt(
                    torch.sum((q_gt - pre_q_norm) * (q_gt - pre_q_norm),
                              dim=-1,
                              keepdim=True) + 1e-10))
            loss_t = torch.mean(
                torch.sqrt((pre_t - t_gt) * (pre_t - t_gt) + 1e-10))
            loss = loss_t * torch.exp(
                -w_x) + w_x + loss_q * torch.exp(
                    -w_q) + w_q
            loss_list.append(weight * loss)
        return sum(loss_list)