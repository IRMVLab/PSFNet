import os
import torch
import numpy as np
import os.path as osp
import torch.utils.data as data

from dataset.sf_preprocess import RangeLimitWithSF, TransformerWithSF, RandomSampleWithSF

__all__ = ['lidarKITTI']


class lidarKITTI(data.Dataset):

    def __init__(self, dataset_path, just_eval, train, pre_processes):
        super().__init__()
        self.root = dataset_path
        self.just_eval = just_eval
        self.train = train
        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

        self._init_pre_processes(pre_processes)

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        data = np.load(self.samples[index])
        data_dict = {'pc1': data['pc1'], 'pc2': data['pc2'], 'flow': data['flow']}
        data_dict = self.apply_pre_processes(data_dict)

        pc1 =  data_dict['pc1'].astype(np.float32)
        pc2 =  data_dict['pc2'].astype(np.float32)
        flow =  data_dict['flow'].astype(np.float32)
        color1 = pc1
        color2 = pc2
        return {'pc1': pc1, 'pc2': pc2, 'norm1': color1, 'norm2': color2, 'flow': flow, 'path': self.samples[index]}


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Number of points per point cloud: {}\n'.format(self.num_points)
        fmt_str += '    is removing ground: {}\n'.format(self.remove_ground)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

    def make_dataset(self):
        root = osp.realpath(osp.expanduser(self.root))
        all_paths = sorted(os.walk(root))
        useful_paths = [os.path.join(all_paths[0][0], file) for file in sorted(all_paths[0][2])]
        if self.just_eval:
            res_paths = useful_paths
        else:
            if self.train:
                res_paths = useful_paths[:100]
            else:
                res_paths = useful_paths[100:]
        return res_paths

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






