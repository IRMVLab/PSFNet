# -- coding: utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F


from pointnet2 import pointnet2_utils

# import pointnet2.pointnet2_utils as pointnet2_utils

LEAKY_RATE = 0.1
use_bn = False

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_activation=True,
                 use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if use_activation:
            relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        else:
            relu = nn.Identity()

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.composed_module(x)
        x = x.permute(0, 2, 1)
        return x


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=[1, 1], bn=False, activation_fn=True):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = bn
        self.activation_fn = activation_fn

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if bn:
            self.bn_linear = nn.BatchNorm2d(out_channels)

        if activation_fn:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x (b,n,s,c)
        # print('x is ')
        # print('x: ', x.device)
        x = x.permute(0, 3, 2, 1)  # (b,c,s,n)
        # print(self.conv)
        outputs = self.conv(x)
        # print('self conv has be carried out')
        if self.bn:
            outputs = self.bn_linear(outputs)

        if self.activation_fn:
            outputs = self.relu(outputs)

        outputs = outputs.permute(0, 3, 2, 1)  # (b,n,s,c)
        return outputs


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx  # [B, S, nsample]


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points



def group(nsample, xyz, points):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)  # [B, N, nsample]
    grouped_xyz = index_points_group(xyz, idx)  # [B, N, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, N, nsample, C]
    if points is not None:
        grouped_points = index_points_group(points, idx)  # (B, N, nsample, D) = (B, N, D) (B, N, nsample)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, N, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm  # [B, N, nsample, C+D]  [B, N, nsample, C]


def grouping(feature, K, src_xyz, q_xyz, use_xyz=False):
    '''
    Input:
        feature: (batch_size, ndataset, c)
        K: neighbor size
        src_xyz: original point xyz (batch_size, ndataset, 3)
        q_xyz: query point xyz (batch_size, npoint, 3)
    Return:
        grouped_xyz: (batch_size, npoint, K,3)
        xyz_diff: (batch_size, npoint,K, 3)
        new_points: (batch_size, npoint,K, c+3) if use_xyz else (batch_size, npoint,K, c)
        point_indices: (batch_size, npoint, K)
    '''
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_xyz = q_xyz.contiguous()
    src_xyz = src_xyz.contiguous()

    point_indices = knn_point(K, src_xyz, q_xyz)  # (batch_size, npoint, K)

    grouped_xyz = index_points_group(src_xyz, point_indices)  # (batch_size, npoint, K,3)
    # print(grouped_xyz.device,q_xyz.device )
    xyz_diff = grouped_xyz - (q_xyz.unsqueeze(2)).repeat(1, 1, K, 1)  # (batch_size, npoint,K, 3)
    # x' - x : KNN points - centroids
    grouped_feature = index_points_group(feature, point_indices)  # (batch_size, npoint, K,c)
    if use_xyz:
        new_points = torch.cat([xyz_diff, grouped_feature], dim=-1)  # (batch_size, npoint,K, c+3)
    else:
        new_points = grouped_feature  # (batch_size, npoint, K,c)

    return grouped_xyz, xyz_diff, new_points, point_indices


def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)  # [B,S,nsample,C]
    grouped_xyz = index_points_group(s_xyz, idx)  # [B, S, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, S, nsample, C]
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)  # [B, S, nsample, D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, S, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


def sample_and_group(sample_idx, npoint, nsample, xyz, points):
    xyz = xyz.contiguous()
    if sample_idx is not None:
        sample_idx = sample_idx
    else:
        sample_idx = pointnet2_utils.furthest_point_sample(xyz, npoint)  # [B, N]
    new_xyz = index_points_gather(xyz, sample_idx)  # [B, N, 3]

    if points is None:
        # (b, n, nsample,3) (b, n,nsample. 3) (b, n, nsample, 3) (b,n,nsample)
        grouped_xyz, xyz_diff, grouped_points, idx = grouping(xyz, nsample, xyz, new_xyz)
    else:
        grouped_xyz, xyz_diff, grouped_points, idx = grouping(points, nsample, xyz, new_xyz)
    new_points = torch.cat([xyz_diff, grouped_points], dim=-1)  # (b, n, nample, 3+c)

    return new_xyz, new_points, sample_idx  # (b, n, 3) (b, n, nsample, 3+c) (b, n)


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit=[8, 8], bn=use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        # xyz : BxCxKxN  [B, C, nsample, N]

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights = F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights  # [B, out_channel, nsample, N]


class PointNetSaModule(nn.Module):
    def __init__(self, npoint: int, radius: float, nsample: int, in_channels: int, mlp: list, mlp2: list,
                 group_all: bool, is_training: bool, bn_decay: bool, bn: bool = True, pooling: str = 'max',
                 knn: bool = False, use_xyz: bool = True):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.in_channels = in_channels + 3
        self.mlp = mlp
        self.mlp2 = mlp2
        self.group_all = group_all
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.num_mlp_layers = len(mlp)
        self.mlp_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.out_channels = mlp[-1]

        for num_out_channel in mlp:
            self.mlp_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

        if mlp2:
            self.out_channels = mlp2[-1]
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
                self.in_channels = num_out_channel

    def forward(self, xyz, feature, sample_idx=None):

        # (b,n,3) (b, n,nsample,3+C] (b, n)
        new_xyz, new_feature, sample_idx = sample_and_group(sample_idx, self.npoint, self.nsample, xyz, feature)

        # new_points: (b, npoint, nample, 3+channel)
        for i, conv in enumerate(self.mlp_convs):
            new_feature = conv(new_feature)
        if self.pooling == 'max':
            new_feature = torch.max(new_feature, dim=2, keepdim=True)[0]  # (b, npoint, 1, mlp[-1])
        elif self.pooling == 'avg':
            new_feature = torch.mean(new_feature, dim=2, keepdim=True)  # (b, npoint, 1, mlp[-1])
        if self.mlp2 is not None:
            for i, conv in enumerate(self.mlp2_convs):
                new_feature = conv(new_feature)
        new_feature = new_feature.squeeze(2)  # (b,npoint, mlp2[-1]) if mlp2 is not None else  (b,npoint, mlp[-1])

        return new_xyz, new_feature, sample_idx


class CostVolume(nn.Module):
    def __init__(self, radius: float, nsample: int, nsample_q: int, in_channels: int, mlp1: list, mlp2: list,
                 is_training: bool, bn_decay: bool, bn: bool = True, pooling: str = 'max',
                 knn: bool = True, corr_func: str = 'elementwise_product', cost_local: bool = False):

        super().__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = 2 * in_channels + 10
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.local = cost_local
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        self.pi_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)
        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        assert mlp1[-1] == mlp2[-1]
        self.out_channels = mlp1[-1]

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.in_channels = 2 * mlp1[-1] + in_channels
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points):
        """

        :param warped_xyz: [B,N,3]
        :param warped_points:  [B,N,C]
        :param f2_xyz: [B,N,3]
        :param f2_points: [B,N,C]
        :return: [B,N,C,mlp2[-1]]
        """

        # [B,N,nsample_q,3] [B,N,nsample_q,3] [B,N,nsample_q,C]
        qi_xyz_grouped, _, qi_points_grouped, idx = grouping(f2_points, self.nsample_q, f2_xyz, warped_xyz)
        pi_xyz_expanded = (torch.unsqueeze(warped_xyz, 2)).repeat([1, 1, self.nsample_q, 1])  # [B,N,nsample_q,3]
        pi_points_expanded = (torch.unsqueeze(warped_points, 2)).repeat([1, 1, self.nsample_q, 1])  # [B,N,nsample_q,C]
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded  # [B,N,nsample_q,3]
        pi_euc_diff = torch.sqrt(torch.sum(torch.mul(pi_xyz_diff, pi_xyz_diff), dim=-1,
                                           keepdim=True) + 1e-20)  # [B,N,nsample_q,1]
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff],
                                       dim=3)  # [B,N,nsample_q,10]
        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped], dim=-1)  # [B,N,nsample_q,2C]
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=3)  # [B,N,nsample_q,10+2C]

        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # [B,N,nsample_q,mlp1[-1]]

        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # [B,N,nsample_q,mlp1[-1]]
        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # [B,N,nsample_q ,2*mlp1[-1]]
        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)  # [B,N,nsample_q ,mlp2[-1]]
        WQ = F.softmax(pi_concat, dim=2)  # [B,N,nsample_q ,mlp2[-1]]
        pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # [B,N,mlp1[-1]]

        # [B,N,nsample,3] [B,N,nsample,3] [B,N,nsample,mlp1[-1]]
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(pi_feat1_new, self.nsample, warped_xyz,
                                                             warped_xyz)
        pc_xyz_new = (torch.unsqueeze(warped_xyz, dim=2)).repeat([1, 1, self.nsample, 1])  # [B,N,nsample,3]
        pc_points_new = (torch.unsqueeze(warped_points, dim=2)).repeat([1, 1, self.nsample, 1])  # [B,N,nsample,C]
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # [B,N,nsample,3]
        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # [B,N,nsample,1]
        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # [B,N,nsample,10]
        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # [B,N,nsample, mlp1[-1]]
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped],
                              dim=-1)  # [B,N,nsample, mlp[-1]+3+mlp[-1]]
        for j, conv in enumerate(self.mlp2_convs_2):
            pc_concat = conv(pc_concat)  # [B,N,nsample, mlp2[-1]]
        WP = F.softmax(pc_concat, dim=2)
        pc_feat1_new = WP * pc_points_grouped  # [B,N,nsample, mlp2[-1]]
        cost_volume_self = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # [B,N, mlp2[-1]]

        if self.local:
            knn_idx_self = knn_point(self.nsample + 32, warped_xyz, warped_xyz)  # (B, N, K)
            costVolume_local = index_points_group(cost_volume_self, knn_idx_self)  # (B, N, K, mlp2[-1])
            costVolume_local = torch.max(costVolume_local, 2)[0]  # (B, N, mlp2[-1])
            return cost_volume_self, costVolume_local
        else:
            return cost_volume_self


class SetUpconvModule(nn.Module):
    def __init__(self, nsample: int, in_channels: list, mlp: list, mlp2: list, is_training: bool,
                 bn_decay: bool = None, bn: bool = True, pooling: str = 'max', radius: float = None, knn: bool = True):
        super(SetUpconvModule, self).__init__()
        self.nsample = nsample
        self.out_channels = in_channels[-1] + 3
        self.mlp = mlp
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.radius = radius
        self.knn = knn
        self.mlp_conv = nn.ModuleList()
        self.mlp2_conv = nn.ModuleList()

        if mlp is not None:
            for i, num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.out_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.out_channels = num_out_channel

        if len(mlp) != 0:
            self.out_channels = mlp[-1] + in_channels[0]
        else:
            self.out_channels = self.out_channels + in_channels[0]

        if mlp2:
            for i, num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(Conv2d(self.out_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
                self.out_channels = num_out_channel
        

    def forward(self, xyz1, xyz2, feat1, feat2):
        """

        :param xyz1:  [B,N1,3]
        :param xyz2:  [B,N2,3]
        :param feat1: [B,N1,C1],features for xyz1 points
        :param feat2: [B,N2,C2],features for xyz2 points
        :return: [B,N1, mlp[-1] or mlp2[-1] or channel1+3]
        """

        # [B,N1,nsample,3], _ , [B,N1,nsample,C2]
        xyz2_grouped, _, feat2_grouped, idx = grouping(feat2, self.nsample, xyz2, xyz1)
        xyz1_expanded = torch.unsqueeze(xyz1, 2)  # [B ,N1, 1, 3]
        xyz_diff = xyz2_grouped - xyz1_expanded  # [B, N1, nsample, 3]
        net = torch.cat([feat2_grouped, xyz_diff], dim=3)  # [B, N1, nsample, C2+3]

        if self.mlp is not None:
            for i, conv in enumerate(self.mlp_conv):
                net = conv(net)
        if self.pooling == 'max':
            feat1_new = torch.max(net, dim=2, keepdim=False)[0]  # [B, N1, mlp[-1]]
        elif self.pooling == 'avg':
            feat1_new = torch.mean(net, dim=2, keepdim=False)  # [B, N1, mlp[-1]]
        if feat1 is not None:
            feat1_new = torch.cat([feat1_new, feat1], dim=2)  # [B, N1, mlp[-1]+C1]
        feat1_new = torch.unsqueeze(feat1_new, 2)  # [B, N1, 1, mlp[-1]+C1]

        if self.mlp2 is not None:
            for i, conv in enumerate(self.mlp2_conv):
                feat1_new = conv(feat1_new)
        feat1_new = torch.squeeze(feat1_new, 2)  # [B, N1, mlp2[-1]]

        return feat1_new


class PointnetFpModule(nn.Module):
    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True, last_mlp_activation=True):
        super(PointnetFpModule, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.last_mlp_activation = last_mlp_activation
        self.mlp_conv = nn.ModuleList()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not (last_mlp_activation):
                activation_fn = False
            else:
                activation_fn = True
            self.mlp_conv.append(
                Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn, activation_fn=activation_fn))
            self.in_channels = num_out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
        """
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist, idx = pointnet2_utils.three_nn(xyz1, xyz2)  # (b,n1,3)
        dist[dist < 1e-10] = 1e-10
        norm = torch.sum((1.0 / dist), dim=2, keepdim=True)
        norm = norm.repeat(1, 1, 3)
        weight = (1.0 / dist) / norm
        points2 = points2.permute(0, 2, 1)
        interpolated_points = pointnet2_utils.three_interpolate(points2.contiguous(), idx, weight)
        interpolated_points = interpolated_points.permute(0, 2, 1)  # (b,n1,c2)

        new_points1 = interpolated_points

        if points1 is not None:
            new_points1 = torch.cat([interpolated_points, points1], dim=2)  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points  # B,ndataset1,nchannel2
        new_points1 = torch.unsqueeze(new_points1, 2)

        for i, conv in enumerate(self.mlp_conv):
            new_points1 = conv(new_points1)

        new_points1 = torch.squeeze(new_points1, 2)  # B,ndataset1,mlp[-1]
        return new_points1


class WarpingLayers(nn.Module):

    def forward(self, xyz1, upsampled_flow):
        return xyz1 + upsampled_flow


class SceneFlow_Estimator(nn.Module):

    def __init__(self, in_channels, flow_ch=3, occ_ch=1, channels=[128, 128], mlp=[128, 64], neighbors=9,
                 clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlow_Estimator, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = in_channels + flow_ch + occ_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = Conv1d(last_channel, 3)

    def forward(self, xyz, feats, feats_new, cost_volume, up_flow=None, occ_mask=None):
        """

        :param xyz: B N 3
        :param feats: B N C1
        :param feats_new: B N C2
        :param cost_volume: B N C3
        :param up_flow: B N 3
        :param occ_mask: B N 1
        :return: new_points: B N  mlp[-1]
        """
        if up_flow is None:
            new_points = torch.cat([feats, feats_new, cost_volume, occ_mask], dim=-1)  # (B, N, C1+C2+C3+1)
        else:
            new_points = torch.cat([feats, feats_new, cost_volume, up_flow, occ_mask], dim=-1)  # (B, N, C1+C2+C3+3+1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)

        if up_flow is not None:
            flow = up_flow + flow

        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class FlowPredictor(nn.Module):

    def __init__(self, in_channels, mlp, is_training, bn_decay, bn=True):
        """

        :param in_channels:
        :param mlp:  in_channels --> mlp[0] --> mlp[-1]
        :param is_training:
        :param bn_decay:
        :param bn:
        """

        super(FlowPredictor, self).__init__()
        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()
        self.out_channels = mlp[-1]

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel

    def forward(self, points_f1: torch.Tensor, upsampled_feat: torch.Tensor, cost_volume: torch.Tensor,
                occ_mask=None) -> torch.Tensor:
        """

        :param points_f1: [B,N,C1]
        :param upsampled_feat: [B,N,C2]
        :param cost_volume: [B,N,C3]
        :return: [B,N,mlp[-1]]
        """
        if occ_mask is not None:
            if upsampled_feat is not None:
                points_concat = torch.cat([points_f1, cost_volume, upsampled_feat, occ_mask], -1)  # [B,N,C1+C2+C3+1]
            else:
                points_concat = torch.cat([points_f1, cost_volume, occ_mask], -1)
        else:
            if upsampled_feat is not None:
                points_concat = torch.cat([points_f1, cost_volume, upsampled_feat], -1)  # [B,N,C1+C2+C3]
            else:
                points_concat = torch.cat([points_f1, cost_volume], -1)

        points_concat = torch.unsqueeze(points_concat, 2)  # [B,N,1,C1+C2+C3]
        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)
        points_concat = torch.squeeze(points_concat, 2)

        return points_concat  # [B,N,mlp[-1]]


class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[1]

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points)  # [B, N, nsample, C+D]  [B, N, nsample, C]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)  # [B, C, nsample, N] =
        weights = self.weightnet(grouped_xyz)  # [B, weightnet, nsample, N]= [B, C, nsample, N]

        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2,
                                                                                              1))  # [B, N, C+D, weightnet] = [B, N, C+D, nsample]*[B, N, nsample, weightnet]
        new_points = new_points.view(B, N, -1)  # [B, N, (C+D) * weightnet]
        new_points = self.linear(new_points)

        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points


class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet=16, bn=use_bn, use_leaky=True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        new_xyz = index_points_gather(xyz, fps_idx)

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz, new_xyz, points)

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other=weights.permute(0, 3, 2, 1)).view(B,
                                                                                                                self.npoint,
                                                                                                                -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.permute(0, 2, 1), new_points, fps_idx


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn=use_bn, use_leaky=True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1

        point_to_patch_cost = torch.sum(weights * new_points, dim=2)  # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)  # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1))  # B C nsample N1
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1),
                                                         knn_idx)  # B, N1, nsample, C
        patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim=2)  # B C N

        return patch_to_patch_cost


class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1=None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1)  # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1)  # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C)  # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim=2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1)  # B 3 N2

        return warped_xyz2


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)
        return dense_flow


class SceneFlowEstimatorPointConv(nn.Module):

    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class Occ_CostVolume(nn.Module):
    def __init__(self, radius: float, nsample: int, nsample_q: int, in_channels: int, mlp1: list, mlp2: list,
                 is_training: bool, bn_decay: bool, is_mask: bool = False, is_bottom: bool = False, bn: bool = True,
                 pooling: str = 'max',
                 knn: bool = True, corr_func: str = 'elementwise_product'):

        super(Occ_CostVolume, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = 2 * in_channels + 10
        self.out_channels = mlp2[-1]
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.mask = is_mask
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_2 = nn.ModuleList()

        self.pi_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)
        self.pc_encoding = Conv2d(10, mlp1[-1], [1, 1], stride=[1, 1], bn=True)

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.in_channels = 2 * mlp1[-1]
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        self.in_channels = 2 * mlp1[-1] + in_channels
        if is_mask:
            if not is_bottom:
                self.in_channels = self.in_channels + 1

        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs_2.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=True))
            self.in_channels = num_out_channel

        if is_mask:
            self.fc_occ = nn.Sequential(
                nn.Conv1d(self.in_channels, 3, 1),
                nn.LeakyReLU(LEAKY_RATE, inplace=True),
                nn.Conv1d(3, 1, 1),
                nn.Sigmoid()
            )

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points, up_occ_mask=None):
        """

        :param warped_xyz: [B,N,3]
        :param warped_points:  [B,N,C]
        :param f2_xyz: [B,N,3]
        :param f2_points: [B,N,C]
        :param up_occ_mask: [B,1,N]
        :return: [B,N,C,mlp2[-1]]
        """
        # [B,N,nsample_q,3] [B,N,nsample_q,3] [B,N,nsample_q,C]
        qi_xyz_grouped, _, qi_points_grouped, idx = grouping(f2_points, self.nsample_q, f2_xyz, warped_xyz)
        pi_xyz_expanded = (torch.unsqueeze(warped_xyz, 2)).repeat([1, 1, self.nsample_q, 1])  # [B,N,nsample_q,3]
        pi_points_expanded = (torch.unsqueeze(warped_points, 2)).repeat([1, 1, self.nsample_q, 1])  # [B,N,nsample_q,C]
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded  # [B,N,nsample_q,3]
        pi_euc_diff = torch.sqrt(torch.sum(torch.mul(pi_xyz_diff, pi_xyz_diff), dim=-1,
                                           keepdim=True) + 1e-20)  # [B,N,nsample_q,1]
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff],
                                       dim=3)  # [B,N,nsample_q,10]
        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped], dim=-1)  # [B,N,nsample_q,2C]
        pi_feat1_new = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=3)  # [B,N,nsample_q,10+2C]

        for i, conv in enumerate(self.mlp1_convs):
            pi_feat1_new = conv(pi_feat1_new)  # [B,N,nsample_q,mlp1[-1]]

        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat)  # [B,N,nsample_q,mlp1[-1]]
        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new], dim=3)  # [B,N,nsample_q ,2*mlp1[-1]]
        for j, conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)  # [B,N,nsample_q ,mlp2[-1]]
        WQ = F.softmax(pi_concat, dim=2)  # [B,N,nsample_q ,mlp2[-1]]
        pi_feat1_new = WQ * pi_feat1_new  # mlp1[-1]=mlp2[-1]
        pi_feat1_new = torch.sum(pi_feat1_new, dim=2, keepdim=False)  # [B,N,mlp1[-1]]

        # [B,N,nsample,3] [B,N,nsample,3] [B,N,nsample,mlp1[-1]]
        pc_xyz_grouped, _, pc_points_grouped, idx = grouping(pi_feat1_new, self.nsample, warped_xyz,
                                                             warped_xyz)
        pc_xyz_new = (torch.unsqueeze(warped_xyz, dim=2)).repeat([1, 1, self.nsample, 1])  # [B,N,nsample,3]
        pc_points_new = (torch.unsqueeze(warped_points, dim=2)).repeat([1, 1, self.nsample, 1])  # [B,N,nsample,C]
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new  # [B,N,nsample,3]
        pc_euc_diff = torch.sqrt(torch.sum(torch.mul(pc_xyz_diff, pc_xyz_diff), dim=3,
                                           keepdim=True) + 1e-20)  # [B,N,nsample,1]
        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff],
                                       dim=3)  # [B,N,nsample,10]
        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)  # [B,N,nsample, mlp1[-1]]
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped], dim=-1)  # [B,N,nsample, 2*mlp[-1]+C]

        if self.mask:
            if up_occ_mask is not None:
                pc_concat = torch.cat(
                    [pc_concat, up_occ_mask.unsqueeze(3).repeat(1, 1, self.nsample, 1)],
                    dim=-1)  # [B,N,nsample, 2*mlp[-1]+C+1]

        for j, conv in enumerate(self.mlp2_convs_2):
            pc_concat = conv(pc_concat)  # [B,N,nsample, mlp2[-1]]
        WP = F.softmax(pc_concat, dim=2)
        pc_feat1_new = WP * pc_points_grouped  # [B,N,nsample, mlp2[-1]]
        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)  # [B,N, mlp2[-1]]

        if self.mask:
            occ_mask = self.fc_occ(pc_feat1_new.permute(0, 2, 1))  # [B, 1, N]
            return occ_mask.permute(0, 2, 1)  # [B, N, 1]

        return pc_feat1_new  # [B,N, mlp2[-1]]


class Occ_weighted_CV(nn.Module):
    def __init__(self, radius: float, nsample: int, nsample_q: int, in_channels: int, mlp1: list, mlp2: list,
                 mlp1_occ: list, mlp2_occ: list, is_training: bool, bn_decay: bool, is_bottom: bool = False,
                 bn: bool = True, pooling: str = 'max', knn: bool = True, corr_func: str = 'elementwise_product'):
        super(Occ_weighted_CV, self).__init__()
        self.nsample = nsample
        self.occ_mask_cv = Occ_CostVolume(radius, nsample, nsample_q, in_channels, mlp1_occ, mlp2_occ,
                                          is_training, bn_decay, True, is_bottom, bn, pooling, knn, corr_func)
        self.cost_self_cv = Occ_CostVolume(radius, nsample, nsample_q, in_channels, mlp1, mlp2,
                                           is_training, bn_decay, False, False, bn, pooling, knn, corr_func)
        self.out_channels = mlp2[-1]

    def forward(self, warped_xyz, warped_points, f2_xyz, f2_points, up_occ_mask=None):
        occ_mask = self.occ_mask_cv(warped_xyz, warped_points, f2_xyz, f2_points, up_occ_mask)  # [B, N, 1]
        costVolume_self = self.cost_self_cv(warped_xyz, warped_points, f2_xyz, f2_points)  # [B, N, mlp2[-1]]
        knn_idx_self = knn_point(self.nsample + 32, warped_xyz, warped_xyz)  # (B, N, K)
        costVolume_local = index_points_group(costVolume_self, knn_idx_self)  # (B, N, K, mlp2[-1])
        costVolume_local = torch.max(costVolume_local, 2)[0]  # (B, N, mlp2[-1])
        costVolume = occ_mask * costVolume_self + (1.0 - occ_mask) * costVolume_local  # (B, N, mlp2[-1])

        return costVolume, occ_mask  # [B, N, mlp2[-1]] [B, N, 1]