# -- coding: utf-8 --

import numpy as np


class BaseRangeLimit(object):
    def __init__(self, x_range=None, y_range=None, z_range=None):
        super().__init__()
        range_list = [x_range, y_range, z_range]
        valid_axis_list = [i for (i, axis_range) in enumerate(range_list) if axis_range is not None]
        self.valid_axis = valid_axis_list
        self.range_list = [sorted(range_list[valid_axis]) for valid_axis in valid_axis_list]

    def __call__(self, pointcloud):
        return self.limit_range(pointcloud)

    def get_limit_range_indices(self, pointcloud):
        mask_list = []
        for valid_axis, axis_range in zip(self.valid_axis, self.range_list):
            mask_list.append(np.logical_and(pointcloud[:, valid_axis] >= axis_range[0],
                                            pointcloud[:, valid_axis] <= axis_range[1]))
        masks = np.ones((pointcloud.shape[0]))
        for mask in mask_list:
            masks = np.logical_and(masks, mask)
        indices = np.where(masks)[0]
        return indices

    def limit_range(self, pointcloud):
        indices = self.get_limit_range_indices(pointcloud)
        return pointcloud[indices]


class BaseRandomSample(object):
    def __init__(self, num_points, allow_less_points):
        super().__init__()
        self.num_points = num_points
        self.allow_less_points = allow_less_points

    def __call__(self, pointcloud):
        return self.random_sample(pointcloud)

    def random_sample(self, pointcloud):
        sample_idx = self.get_random_sample_indices(pointcloud.shape[0])
        return pointcloud[sample_idx]

    def get_random_sample_indices(self, N):
        if N >= self.num_points:
            sample_idx = np.random.choice(N, self.num_points, replace=False)
        else:
            if self.allow_less_points:
                return np.arange(N)
            else:
                repeat_times = int(self.num_points / N)
                sample_num = self.num_points % N
                sample_idx = np.concatenate(
                    [np.repeat(np.arange(N), repeat_times), np.random.choice(N, sample_num, replace=False)],
                    axis=-1)
        return sample_idx


class BaseTransform(object):
    def __init__(self):
        super().__init__()

    def transformer_tr(self, pointcloud, Tr):
        Tr = np.array(Tr)
        if Tr.size != 16:
            raise ValueError(f'the size of Tr your input is {Tr.size}, but expected 16')
        self.Tr = np.array(Tr).reshape(4, 4)
        pointcloud = np.concatenate([pointcloud[:, :3], np.ones((pointcloud.shape[0], 1))], axis=-1)
        pointcloud = pointcloud @ Tr.T
        pointcloud = pointcloud[:, :3]
        return pointcloud
