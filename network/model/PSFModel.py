# -*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.pointconv_util import Conv1d, FlowPredictor, WarpingLayers, PointNetSaModule, Occ_weighted_CV, CostVolume, \
    Occ_CostVolume, SetUpconvModule, PointnetFpModule


class PSFModel(nn.Module):
    def __init__(self, bn_decay=None):
        super().__init__()
        RADIUS1 = 0.5
        RADIUS2 = 1.0
        RADIUS3 = 2.0
        RADIUS4 = 4.0

        is_training = self.training 
        self.w_x = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.w_q = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True) 

        self.layer0 = PointNetSaModule(npoint=2048, radius=RADIUS1, nsample=32, in_channels=3, mlp=[8, 8, 16],
                                       mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer1 = PointNetSaModule(npoint=1024, radius=RADIUS1, nsample=32, in_channels=16, mlp=[16, 16, 32],
                                       mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer2 = PointNetSaModule(npoint=256, radius=RADIUS2, nsample=16, in_channels=32, mlp=[32, 32, 64],
                                       mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)
        self.layer3 = PointNetSaModule(npoint=64, radius=RADIUS3, nsample=16, in_channels=64, mlp=[64, 64, 128],
                                       mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)


        self.warping = WarpingLayers()
        self.fp = PointnetFpModule(in_channels=3, mlp=[], is_training=is_training, bn_decay=bn_decay)
        self.mask_fp = PointnetFpModule(in_channels=1, mlp=[], is_training=is_training, bn_decay=bn_decay)
        self.occ_mask_fp = PointnetFpModule(in_channels=1, mlp=[], is_training=is_training, bn_decay=bn_decay)


        self.cost_flow = Occ_weighted_CV(radius=10.0, nsample=4, nsample_q=6, in_channels=128, mlp1=[256, 128, 128],
                                         mlp2=[256, 128], mlp1_occ=[256, 128], mlp2_occ=[128], is_training=is_training,
                                         bn_decay=bn_decay, is_bottom=True, bn=True, pooling='max', knn=True,
                                         corr_func='concat')

        self.occ_mask_cost2 = Occ_CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=64, mlp1=[128, 64],
                                             mlp2=[64], is_training=is_training,
                                             bn_decay=bn_decay, is_mask=True, bn=True, pooling='max', knn=True,
                                             corr_func='concat')
        self.occ_mask_cost1 = Occ_CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=32, mlp1=[128, 64],
                                             mlp2=[64], is_training=is_training,
                                             bn_decay=bn_decay, is_mask=True, bn=True, pooling='max', knn=True,
                                             corr_func='concat')
        self.occ_mask_cost0 = Occ_CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=16,
                                             mlp1=[128, 64],
                                             mlp2=[64], is_training=is_training,
                                             bn_decay=bn_decay, is_mask=True, bn=True, pooling='max', knn=True,
                                             corr_func='concat')

        self.layer3_costSa = PointNetSaModule(npoint=64, radius=RADIUS3, nsample=16, in_channels=64, mlp=[64, 64, 128],
                                              mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)

        self.layer4_costSa = PointNetSaModule(npoint=16, radius=RADIUS4, nsample=8, in_channels=128,
                                              mlp=[128, 128, 256],
                                              mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)

        self.upconv1 = SetUpconvModule(nsample=8, radius=2.4, in_channels=[128, 256], mlp=[128, 128, 256], mlp2=[256],
                                       is_training=is_training, bn_decay=bn_decay, knn=True)

        self.upconv2 = SetUpconvModule(nsample=8, radius=1.2, in_channels=[64, 128], mlp=[128, 64, 64], mlp2=[64],
                                       is_training=is_training, bn_decay=bn_decay, knn=True)
        self.upconv3 = SetUpconvModule(nsample=8, radius=1.2, in_channels=[32, 64], mlp=[128, 64, 64], mlp2=[64],
                                       is_training=is_training, bn_decay=bn_decay, knn=True)
        self.upconv4 = SetUpconvModule(nsample=8, radius=1.2, in_channels=[16, 64], mlp=[128, 64, 64], mlp2=[64],
                                       is_training=is_training, bn_decay=bn_decay, knn=True)

        self.flow_pred1 = FlowPredictor(in_channels=128 * 2 + 256 + 1, mlp=[256, 128, 128], is_training=is_training,
                                        bn_decay=bn_decay)
        self.flow_pred2 = FlowPredictor(in_channels=64 * 3 + 1, mlp=[128, 64, 64], is_training=is_training,
                                        bn_decay=bn_decay)
        self.flow_pred3 = FlowPredictor(in_channels=64 * 2 + 32 + 1, mlp=[128, 64, 64], is_training=is_training,
                                        bn_decay=bn_decay)
        self.flow_pred4 = FlowPredictor(in_channels=64 * 2 + 16 + 1, mlp=[128, 64, 64], is_training=is_training,
                                        bn_decay=bn_decay)

        self.conv0 = Conv1d(256, 3)
        self.conv1 = Conv1d(128, 3)
        self.conv2 = Conv1d(64, 3)
        self.conv3 = Conv1d(64, 3)
        self.conv4 = Conv1d(64, 3)


        self.cost_pose = CostVolume(radius=10.0, nsample=4, nsample_q=32, in_channels=64, mlp1=[128, 64, 64],
                                    mlp2=[128, 64], is_training=is_training,
                                    bn_decay=bn_decay, bn=True, pooling='max', knn=True,
                                    corr_func='concat')
        self.costSa_pose = PointNetSaModule(npoint=64, radius=RADIUS3, nsample=16, in_channels=64,
                                            mlp=[128, 64, 64],
                                            mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay)


        self.share_cost2 = CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=64, mlp1=[128, 64, 64],
                                      mlp2=[128, 64], is_training=is_training,
                                      bn_decay=bn_decay, bn=True, pooling='max', knn=True,
                                      corr_func='concat', cost_local=True)
        self.share_cost1 = CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=32, mlp1=[128, 64, 64],
                                      mlp2=[128, 64], is_training=is_training,
                                      bn_decay=bn_decay, bn=True, pooling='max', knn=True,
                                      corr_func='concat', cost_local=True)

        self.share_cost0 = CostVolume(radius=10.0, nsample=4, nsample_q=6, in_channels=16, mlp1=[128, 64, 64],
                                      mlp2=[128, 64], is_training=is_training,
                                      bn_decay=bn_decay, bn=True, pooling='max', knn=True,
                                      corr_func='concat', cost_local=True)

        self.upconv2_w = SetUpconvModule(nsample=8, radius=2.4, in_channels=[64, 64], mlp=[128, 64],
                                         mlp2=[64], is_training=is_training,
                                         bn_decay=bn_decay, knn=True)
        self.upconv2_upsample = SetUpconvModule(nsample=8, radius=2.4, in_channels=[64, 64], mlp=[128, 64],
                                                mlp2=[64], is_training=is_training,
                                                bn_decay=bn_decay, knn=True)
        self.upconv3_w = SetUpconvModule(nsample=8, radius=2.4, in_channels=[32, 64], mlp=[128, 64],
                                         mlp2=[64], is_training=is_training,
                                         bn_decay=bn_decay, knn=True)
        self.upconv3_upsample = SetUpconvModule(nsample=8, radius=2.4, in_channels=[32, 64], mlp=[128, 64],
                                                mlp2=[64], is_training=is_training,
                                                bn_decay=bn_decay, knn=True)
        self.upconv4_w = SetUpconvModule(nsample=8, radius=2.4, in_channels=[16, 64], mlp=[128, 64],
                                         mlp2=[64], is_training=is_training,
                                         bn_decay=bn_decay, knn=True)
        self.upconv4_upsample = SetUpconvModule(nsample=8, radius=2.4, in_channels=[16, 64], mlp=[128, 64],
                                                mlp2=[64], is_training=is_training,
                                                bn_decay=bn_decay, knn=True)

        self.flow_predictor0 = FlowPredictor(in_channels=128 + 64, mlp=[128, 64], is_training=is_training,
                                             bn_decay=bn_decay)
        self.flow_predictor1_predict = FlowPredictor(in_channels=64 * 3, mlp=[128, 64], is_training=is_training,
                                                     bn_decay=bn_decay)
        self.flow_predictor1_w = FlowPredictor(in_channels=64 * 3, mlp=[128, 64], is_training=is_training,
                                               bn_decay=bn_decay)
        self.flow_predictor2_predict = FlowPredictor(in_channels=64 * 2 + 32, mlp=[128, 64], is_training=is_training,
                                                     bn_decay=bn_decay)
        self.flow_predictor2_w = FlowPredictor(in_channels=64 * 2 + 32, mlp=[128, 64], is_training=is_training,
                                               bn_decay=bn_decay)
        self.flow_predictor3_predict = FlowPredictor(in_channels=64 * 2 + 16, mlp=[128, 64], is_training=is_training,
                                                     bn_decay=bn_decay)
        self.flow_predictor3_w = FlowPredictor(in_channels=64 * 2 + 16, mlp=[128, 64], is_training=is_training,
                                               bn_decay=bn_decay)

        self.conv_static_l3 = nn.Sequential(
            Conv1d(64, 1, use_activation=False),
            nn.Sigmoid())
        self.conv_static_l2 = nn.Sequential(
            Conv1d(64, 1, use_activation=False),
            nn.Sigmoid())
        self.conv_static_l1 = nn.Sequential(
            Conv1d(64, 1, use_activation=False),
            nn.Sigmoid())
        self.conv_static_l0 = nn.Sequential(
            Conv1d(64, 1, use_activation=False),
            nn.Sigmoid())

        self.conv_q_l3 = Conv1d(256, 4, use_activation=False)
        self.conv_t_l3 = Conv1d(256, 3, use_activation=False)
        self.conv_drop_l3 = Conv1d(64, 256, use_activation=False)

        self.conv_q_l2 = Conv1d(256, 4, use_activation=False)
        self.conv_t_l2 = Conv1d(256, 3, use_activation=False)
        self.conv_drop_l2 = Conv1d(64, 256, use_activation=False)

        self.conv_q_l1 = Conv1d(256, 4, use_activation=False)
        self.conv_t_l1 = Conv1d(256, 3, use_activation=False)
        self.conv_drop_l1 = Conv1d(64, 256, use_activation=False)

        self.conv_q_l0 = Conv1d(256, 4, use_activation=False)
        self.conv_t_l0 = Conv1d(256, 3, use_activation=False)
        self.conv_drop_l0 = Conv1d(64, 256, use_activation=False)

    def forward(self, xyz1, xyz2, color1, color2, state='pose', label=None):

        batch_size = xyz1.shape[0]

        l0_xyz_f1_raw = xyz1
        l0_xyz_f1 = xyz1
        l0_points_f1 = color1

        l0_xyz_f2_raw = xyz2
        l0_xyz_f2 = xyz2
        l0_points_f2 = color2

        if label is None:
            label = torch.zeros(xyz1.size(), device='cuda')
        l0_label_f1 = label

        l0_xyz_f1, l0_label_f1, l0_points_f1, pc1_sample_2048, pc1_idx_2048 = self.layer0(l0_xyz_f1, l0_xyz_f1_raw,
                                                                                          l0_label_f1,
                                                                                          l0_points_f1)

        l1_xyz_f1, l1_label, l1_points_f1, pc1_sample_1024, pc1_idx_1024 = self.layer1(l0_xyz_f1,
                                                                                       pc1_sample_2048,
                                                                                       l0_label_f1,
                                                                                       l0_points_f1)

        l2_xyz_f1, l2_label, l2_points_f1, pc1_sample_256, pc1_idx_256 = self.layer2(l1_xyz_f1, pc1_sample_1024,
                                                                                     l1_label,
                                                                                     l1_points_f1)

        l3_xyz_f1, l3_label, l3_points_f1, pc1_sample_64, pc1_idx_64 = self.layer3(l2_xyz_f1, pc1_sample_256,
                                                                                   l2_label, l2_points_f1)

        l0_xyz_f2, _, l0_points_f2, pc2_sample, pc2_idx_2048 = self.layer0(l0_xyz_f2, l0_xyz_f2_raw, label,
                                                                           l0_points_f2)

        l1_xyz_f2, _, l1_points_f2, pc2_idx_1024 = self.layer1(l0_xyz_f2, None, l0_label_f1,
                                                               l0_points_f2)

        l2_xyz_f2, _, l2_points_f2, pc2_idx_256 = self.layer2(l1_xyz_f2, None, l1_label,
                                                              l1_points_f2)

        l3_xyz_f2, _, l3_points_f2, pc2_idx_64 = self.layer3(l2_xyz_f2, None, l2_label,
                                                             l2_points_f2)

        l2_cost_f1 = self.cost_pose(l2_xyz_f1, l2_points_f1, l2_xyz_f2, l2_points_f2)

        l3_xyz_f1_pose, l3_label_pose, l3_cost_f1_pose, _ = self.costSa_pose(l2_xyz_f1, None, l2_label,
                                                                             l2_cost_f1,
                                                                             sample_idx=pc1_idx_64)

        l3_xyz_f1, l3_label, l3_cost_f1, _ = self.layer3_costSa(l2_xyz_f1, None, l2_label,
                                                                l2_cost_f1,
                                                                sample_idx=pc1_idx_64)

        l4_xyz_f1, _, l4_cost_f1, _ = self.layer4_costSa(l3_xyz_f1, None, l3_label,
                                                         l3_cost_f1)

        l3_cost_f1_new = self.upconv1(l3_xyz_f1, l4_xyz_f1, l3_cost_f1,
                                      l4_cost_f1)

        l3_flow_coarse = self.conv0(l3_cost_f1_new)  
        l3_flow_warped = self.warping(l3_xyz_f1, l3_flow_coarse)  
        l3_cost_volume_flow, occ_mask3 = self.cost_flow(l3_flow_warped, l3_cost_f1, l3_xyz_f2,
                                                        l3_points_f2) 
        l3_flow_finer = self.flow_pred1(l3_cost_f1, l3_cost_f1_new,
                                        l3_cost_volume_flow,
                                        occ_mask3) 
        l3_flow_det = self.conv1(l3_flow_finer)
        l3_flow = l3_flow_coarse + l3_flow_det  

        l3_embedded_feature = self.flow_predictor0(l3_points_f1, None,
                                                   l3_cost_f1_pose)  
        l3_W_static_feature = F.softmax(l3_embedded_feature, dim=1)  
        l3_points_f1_new_sum = torch.sum(l3_cost_f1_pose * l3_W_static_feature, (1,), keepdim=True) 
        l3_static_mask = self.conv_static_l3(l3_W_static_feature)

        l3_q_t = self.conv_drop_l3(l3_points_f1_new_sum)  
        l3_q_t_drop = F.dropout(l3_q_t, p=0.5, training=self.training)  
        l3_q_coarse = self.conv_q_l3(l3_q_t_drop)
        l3_q_coarse = l3_q_coarse / (
                torch.sqrt(torch.sum(l3_q_coarse * l3_q_coarse, dim=-1, keepdim=True) + 1e-10) + 1e-10) 
        l3_t_coarse = self.conv_t_l3(l3_q_t_drop) 
        l3_q = torch.squeeze(l3_q_coarse, dim=1)
        l3_t = torch.squeeze(l3_t_coarse, dim=1)

        l2_q_coarse = torch.reshape(l3_q, [batch_size, 1, -1])
        l2_t_coarse = torch.reshape(l3_t, [batch_size, 1, -1])
        l2_q_inv = inv_q(l2_q_coarse, batch_size)
        pc1_sample_256_q = torch.cat([torch.zeros([batch_size, 256, 1]).cuda(), l2_xyz_f1], dim=-1)
        l2_pose_warped = mul_q_point(l2_q_coarse, pc1_sample_256_q, batch_size)
        l2_pose_warped = torch.index_select(mul_point_q(l2_pose_warped, l2_q_inv, batch_size), 2,
                                            torch.LongTensor(range(1, 4)).cuda()) + l2_t_coarse

        l2_flow_coarse = self.fp(l2_xyz_f1, l3_xyz_f1, None, l3_flow) 
        l2_flow_warped = self.warping(l2_xyz_f1, l2_flow_coarse) 

        l2_static_up_mask = self.mask_fp(l2_xyz_f1, l3_xyz_f1, None, l3_static_mask)  
        up_occ_mask2 = self.occ_mask_fp(l2_xyz_f1, l3_xyz_f1, None, occ_mask3) 
        l2_warp_mask = F.softmax(l2_static_up_mask * (1 - up_occ_mask2), dim=1) 

        if state == 'pose':
            l2_xyz_warped = l2_pose_warped
        elif state == 'flow':
            l2_xyz_warped = l2_flow_warped
        elif state == 'all':
            l2_xyz_warped = l2_warp_mask * l2_pose_warped + (1 - l2_warp_mask) * l2_flow_warped
        else:
            raise ValueError(" state input is wrong! please input 'pose','flow' or 'all' ")

        l2_cost_volume_self, l2_cost_volume_local = self.share_cost2(l2_xyz_warped, l2_points_f1, l2_xyz_f2,
                                                                     l2_points_f2)

        occ_mask2 = self.occ_mask_cost2(l2_xyz_warped, l2_points_f1, l2_xyz_f2, l2_points_f2, up_occ_mask2)

        l2_cost_volume_flow = l2_cost_volume_self * occ_mask2 + (1 - occ_mask2) * l2_cost_volume_local

        l2_points_f1_new = self.upconv2(l2_xyz_f1, l3_xyz_f1, l2_points_f1,
                                        l3_flow_finer)


        l2_flow_finer = self.flow_pred2(l2_points_f1, l2_points_f1_new,
                                        l2_cost_volume_flow,
                                        occ_mask2)
        l2_flow_det = self.conv2(l2_flow_finer)  
        l2_flow = l2_flow_coarse + l2_flow_det  


        l2_embedded_feature_upsample = self.upconv2_w(l2_xyz_f1, l3_xyz_f1, l2_points_f1,
                                                      l3_embedded_feature)
        l2_cost_volume_upsample = self.upconv2_upsample(l2_xyz_f1, l3_xyz_f1, l2_points_f1,
                                                        l3_cost_f1_pose)

        l2_cost_volume_pose = self.flow_predictor1_predict(l2_points_f1, l2_cost_volume_upsample,
                                                           l2_cost_volume_self)
        l2_embedded_feature = self.flow_predictor1_w(l2_points_f1, l2_embedded_feature_upsample,
                                                     l2_cost_volume_pose) 
        l2_W_static_feature = F.softmax(l2_embedded_feature, dim=1)  
        l2_static_mask = self.conv_static_l2(l2_W_static_feature)

        l2_cost_volume_sum = torch.sum(l2_cost_volume_pose * l2_W_static_feature, dim=1,
                                       keepdim=True)  
        l2_q_t = self.conv_drop_l2(l2_cost_volume_sum)  
        l2_q_t_drop = F.dropout(l2_q_t, p=0.5, training=self.training)
        l2_q_det = self.conv_q_l2(l2_q_t_drop) 
        l2_q_det = l2_q_det / (torch.sqrt(torch.sum(l2_q_det * l2_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l2_t_det = self.conv_t_l2(l2_q_t_drop)

        l2_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l2_t_coarse], dim=-1)
        l2_t_coarse_trans = mul_q_point(l2_q_det, l2_t_coarse_trans, batch_size)
        l2_t_coarse_trans = torch.index_select(mul_point_q(l2_t_coarse_trans, l2_q_inv, batch_size), 2,
                                               torch.LongTensor(range(1, 4)).cuda())

        l2_q = torch.squeeze(mul_point_q(l2_q_det, l2_q_coarse, batch_size), dim=1)
        l2_t = torch.squeeze(l2_t_coarse_trans + l2_t_det, dim=1)

        l1_q_coarse = torch.reshape(l2_q, [batch_size, 1, -1])
        l1_t_coarse = torch.reshape(l2_t, [batch_size, 1, -1])
        l1_q_inv = inv_q(l1_q_coarse, batch_size)
        pc1_sample_1024_q = torch.cat([torch.zeros(batch_size, 1024, 1).cuda(), l1_xyz_f1], dim=-1)
        l1_pose_warped = mul_q_point(l1_q_coarse, pc1_sample_1024_q, batch_size)
        l1_pose_warped = torch.index_select(mul_point_q(l1_pose_warped, l1_q_inv, batch_size), 2,
                                            torch.LongTensor(range(1, 4)).cuda()) + l1_t_coarse

        l1_flow_coarse = self.fp(l1_xyz_f1, l2_xyz_f1, None, l2_flow) 
        l1_flow_warped = self.warping(l1_xyz_f1, l1_flow_coarse) 

        l1_static_up_mask = self.mask_fp(l1_xyz_f1, l2_xyz_f1, None, l2_static_mask)  
        up_occ_mask1 = self.occ_mask_fp(l1_xyz_f1, l2_xyz_f1, None, occ_mask2)
        l1_warp_mask = F.softmax(l1_static_up_mask * (1 - up_occ_mask1), dim=1)

        if state == 'pose':
            l1_xyz_warped = l1_pose_warped
        elif state == 'flow':
            l1_xyz_warped = l1_flow_warped
        elif state == 'all':
            l1_xyz_warped = l1_warp_mask * l1_pose_warped + (1 - l1_warp_mask) * l1_flow_warped
        else:
            raise ValueError(" state input is wrong! please input 'pose','flow' or 'all' ")

        l1_cost_volume_self, l1_cost_volume_local = self.share_cost1(l1_xyz_warped, l1_points_f1, l1_xyz_f2,
                                                                     l1_points_f2)

        occ_mask1 = self.occ_mask_cost1(l1_xyz_warped, l1_points_f1, l1_xyz_f2,
                                        l1_points_f2, up_occ_mask1)

        l1_cost_volume_flow = occ_mask1 * l1_cost_volume_self + (1 - occ_mask1) * l1_cost_volume_local

        l1_points_f1_new = self.upconv3(l1_xyz_f1, l2_xyz_f1, l1_points_f1,
                                        l2_flow_finer)
        l1_flow_finer = self.flow_pred3(l1_points_f1, l1_points_f1_new,
                                        l1_cost_volume_flow,
                                        occ_mask1) 
        l1_flow_det = self.conv3(l1_flow_finer)  
        l1_flow = l1_flow_coarse + l1_flow_det  

        l1_embedded_feature_upsample = self.upconv3_w(l1_xyz_f1, l2_xyz_f1, l1_points_f1,
                                                      l2_embedded_feature)

        l1_cost_volume_upsample = self.upconv3_upsample(l1_xyz_f1, l2_xyz_f1, l1_points_f1,
                                                        l2_cost_volume_pose)

        l1_cost_volume_pose = self.flow_predictor2_predict(l1_points_f1, l1_cost_volume_upsample,
                                                           l1_cost_volume_self)

        l1_embedded_feature = self.flow_predictor2_w(l1_points_f1, l1_embedded_feature_upsample,
                                                     l1_cost_volume_pose)
        l1_W_static_feature = F.softmax(l1_embedded_feature, dim=1)  
        l1_static_mask = self.conv_static_l1(l1_W_static_feature)

        l1_cost_volume_sum = torch.sum(l1_cost_volume_pose * l1_W_static_feature, dim=1,
                                       keepdim=True) 
        l1_q_t = self.conv_drop_l1(l1_cost_volume_sum)  
        l1_q_t_drop = F.dropout(l1_q_t, p=0.5, training=self.training) 
        l1_q_det = self.conv_q_l1(l1_q_t_drop)  
        l1_q_det = l1_q_det / (torch.sqrt(torch.sum(l1_q_det * l1_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l1_t_det = self.conv_t_l1(l1_q_t_drop) 

        l1_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l1_t_coarse], dim=-1)
        l1_t_coarse_trans = mul_q_point(l1_q_det, l1_t_coarse_trans, batch_size)
        l1_t_coarse_trans = torch.index_select(mul_point_q(l1_t_coarse_trans, l1_q_inv, batch_size), 2,
                                               torch.LongTensor(range(1, 4)).cuda())

        l1_q = torch.squeeze(mul_point_q(l1_q_det, l1_q_coarse, batch_size), dim=1)
        l1_t = torch.squeeze(l1_t_coarse_trans + l1_t_det, dim=1)

        l0_q_coarse = torch.reshape(l1_q, [batch_size, 1, -1])
        l0_t_coarse = torch.reshape(l1_t, [batch_size, 1, -1])
        l0_q_inv = inv_q(l0_q_coarse, batch_size)
        pc1_sample_2048_q = torch.cat([torch.zeros([batch_size, 2048, 1]).cuda(), l0_xyz_f1], dim=-1)
        l0_pose_warped = mul_q_point(l0_q_coarse, pc1_sample_2048_q, batch_size)
        l0_pose_warped = torch.index_select(mul_point_q(l0_pose_warped, l0_q_inv, batch_size), 2,
                                            torch.LongTensor(range(1, 4)).cuda()) + l0_t_coarse 

        l0_flow_coarse = self.fp(l0_xyz_f1, l1_xyz_f1, None, l1_flow)
        l0_flow_warped = self.warping(l0_xyz_f1, l0_flow_coarse)

        l0_static_up_mask = self.mask_fp(l0_xyz_f1, l1_xyz_f1, None, l1_static_mask) 
        up_occ_mask0 = self.occ_mask_fp(l0_xyz_f1, l1_xyz_f1, None, occ_mask1)  
        l0_warp_mask = F.softmax(l0_static_up_mask * (1 - up_occ_mask0), dim=1)

        if state == 'pose':
            l0_xyz_warped = l0_pose_warped
        elif state == 'flow':
            l0_xyz_warped = l0_flow_warped
        elif state == 'all':
            l0_xyz_warped = l0_warp_mask * l0_pose_warped + (1 - l0_warp_mask) * l0_flow_warped


        l0_cost_volume_self, l0_cost_volume_local = self.share_cost0(l0_xyz_warped, l0_points_f1, l0_xyz_f2,
                                                                     l0_points_f2)

        occ_mask0 = self.occ_mask_cost0(l0_xyz_warped, l0_points_f1, l0_xyz_f2,
                                        l0_points_f2,
                                        up_occ_mask0)
        l0_cost_volume_flow = occ_mask0 * l0_cost_volume_self + (1 - occ_mask0) * l0_cost_volume_local
        l0_points_f1_new = self.upconv4(l0_xyz_f1, l1_xyz_f1, l0_points_f1,
                                        l1_flow_finer)

        l0_flow_finer = self.flow_pred4(l0_points_f1, l0_points_f1_new,
                                        l0_cost_volume_flow, occ_mask0)
        l0_flow_det = self.conv4(l0_flow_finer) 
        l0_flow = l0_flow_coarse + l0_flow_det  

        l0_embedded_feature_upsample = self.upconv4_w(l0_xyz_f1, l1_xyz_f1, l0_points_f1,
                                                      l1_embedded_feature)


        l0_cost_volume_upsample = self.upconv4_upsample(l0_xyz_f1, l1_xyz_f1, l0_points_f1,
                                                        l1_cost_volume_pose)


        l0_cost_volume_pose = self.flow_predictor3_predict(l0_points_f1, l0_cost_volume_upsample,
                                                           l0_cost_volume_self)

        l0_embedded_feature = self.flow_predictor3_w(l0_points_f1, l0_embedded_feature_upsample,
                                                     l0_cost_volume_pose)
        l0_W_static_feature = F.softmax(l0_embedded_feature, dim=1)
        l0_static_mask = self.conv_static_l0(l0_W_static_feature)

        l0_cost_volume_sum = torch.sum(l0_cost_volume_pose * l0_W_static_feature, dim=1,
                                       keepdim=True) 
        l0_q_t = self.conv_drop_l0(l0_cost_volume_sum) 
        l0_q_t_drop = F.dropout(l0_q_t, p=0.5, training=self.training) 
        l0_q_det = self.conv_q_l0(l0_q_t_drop) 
        l0_q_det = l0_q_det / (torch.sqrt(torch.sum(l0_q_det * l0_q_det, dim=-1, keepdim=True) + 1e-10) + 1e-10)
        l0_t_det = self.conv_t_l0(l0_q_t_drop)  

        l0_t_coarse_trans = torch.cat([torch.zeros([batch_size, 1, 1]).cuda(), l0_t_coarse], dim=-1)
        l0_t_coarse_trans = mul_q_point(l0_q_det, l0_t_coarse_trans, batch_size)
        l0_t_coarse_trans = torch.index_select(mul_point_q(l0_t_coarse_trans, l0_q_inv, batch_size), 2,
                                               torch.LongTensor(range(1, 4)).cuda())
        l0_q = torch.squeeze(mul_point_q(l0_q_det, l0_q_coarse, batch_size), dim=1)
        l0_t = torch.squeeze(l0_t_coarse_trans + l0_t_det, dim=1)


        pre_flow_list = [l0_flow, l1_flow, l2_flow, l3_flow] 
        label = [l0_label_f1, l1_label, l2_label, l3_label]
        pc1_list = [l0_xyz_f1, l1_xyz_f1, l2_xyz_f1, l3_xyz_f1]  
        pc2_list = [l0_xyz_f2, l1_xyz_f2, l2_xyz_f2, l3_xyz_f2]  
        occ_masks = [occ_mask0, occ_mask1, occ_mask2, occ_mask3]  
        static_masks = [l0_static_mask, l1_static_mask, l2_static_mask, l3_static_mask] 
        l0_q_norm = q2qnorm(l0_q)
        l1_q_norm = q2qnorm(l1_q)
        l2_q_norm = q2qnorm(l2_q)
        l3_q_norm = q2qnorm(l3_q)

        pre_q_list = [l0_q_norm, l1_q_norm, l2_q_norm, l3_q_norm]
        pre_t_list = [l0_t, l1_t, l2_t, l3_t]

        return pre_q_list, pre_t_list, pc1_sample_2048, pc1_list, pc2_list, pre_flow_list, occ_masks, static_masks, label


def q2qnorm(input_q):
    q_norm = input_q / (torch.sqrt(torch.sum(input_q * input_q, dim=-1, keepdim=True) + 1e-10) + 1e-10)
    return q_norm


def inv_q(q, batch_size):
    q = torch.squeeze(q, dim=1)
    q_2 = torch.sum(q * q, dim=-1, keepdim=True) + 1e-10
    q0 = torch.index_select(q, 1, torch.LongTensor([0]).cuda())
    q_ijk = -torch.index_select(q, 1, torch.LongTensor([1, 2, 3]).cuda())
    q_ = torch.cat([q0, q_ijk], dim=-1)
    q_inv = q_ / q_2
    return q_inv


def mul_q_point(q_a, q_b, batch_size):
    q_a = torch.reshape(q_a, [batch_size, 1, 4])
    index = [1, 0, 3, 2]
    q_result = []
    for i in range(4):
        q_result_i = torch.mul(q_a[:, :, 0], q_b[:, :, i]) - torch.mul(q_a[:, :, 1], q_b[:, :, index[i]]) - torch.mul(
            q_a[:, :, 2], q_b[:, :, 3 - index[i]]) - torch.mul(q_a[:, :, 3], q_b[:, :, 3 - i])
        q_result_i = torch.reshape(q_result_i, [batch_size, -1, 1])
        q_result.append(q_result_i)
    q_result = torch.cat(q_result, dim=-1)
    return q_result  # B N 4


def mul_point_q(q_a, q_b, batch_size):
    q_b = torch.reshape(q_b, [batch_size, 1, 4])
    index = [1, 0, 3, 2]
    q_result = []
    for i in range(4):
        q_result_i = torch.mul(q_a[:, :, 0], q_b[:, :, i]) - torch.mul(q_a[:, :, 1], q_b[:, :, index[i]]) - torch.mul(
            q_a[:, :, 2], q_b[:, :, 3 - index[i]]) - torch.mul(q_a[:, :, 3], q_b[:, :, 3 - i])
        q_result_i = torch.reshape(q_result_i, [batch_size, -1, 1])
        q_result.append(q_result_i)

    q_result = torch.cat(q_result, dim=-1)
    return q_result  # B N 4
