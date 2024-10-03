import os
import numpy as np
import os.path as osp
import torch.utils.data as data

from dataset.sf_preprocess import RangeLimitWithSF, TransformerWithSF, RandomSampleWithSF

__all__ = ['KSF142']


class KSF142(data.Dataset):
    """
    Args:
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        transform (callable):
        gen_func (callable):
        args:
    """

    def __init__(self, dataset_path, just_eval, train, pre_processes):
        super().__init__()
        self.root = dataset_path
        self.just_eval = just_eval
        self.train = train

        # 构建点云数据预处理
        self._init_pre_processes(pre_processes)
        self.samples = self.make_dataset()
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pc1, pc2 = self.pc_loader(self.samples[index])
        flow = pc2[:, :3] - pc1[:, :3]
        data_dict = {'pc1': pc1, 'pc2': pc2, 'flow': flow}
        data_dict = self.apply_pre_processes(data_dict)
        pc1 = data_dict['pc1'].astype(np.float32)
        pc2 = data_dict['pc2'].astype(np.float32)
        flow = data_dict['flow'].astype(np.float32)
        pc1_norm = pc1
        pc2_norm = pc1
        if pc1 is None:
            print('path {} get pc1 is None'.format(self.samples[index]), flush=True)
            index = np.random.choice(range(self.__len__()))
            return self.__getitem__(index)
        return {'pc1': pc1, 'pc2': pc2, 'norm1': pc1_norm, 'norm2': pc2_norm, 'flow': flow,
                'path': self.samples[index]}


    def make_dataset(self):
        do_mapping = True
        root = osp.realpath(osp.expanduser(self.root))
        all_paths = sorted(os.walk(root))
        useful_paths = [item[0] for item in all_paths if len(item[1]) == 0]
        try:
            assert (len(useful_paths) == 200)
        except AssertionError:
            print('assert (len(useful_paths) == 200) failed!', len(useful_paths))
        if do_mapping:
            mapping_path = osp.join(osp.dirname(__file__), 'KITTI_mapping.txt')
            with open(mapping_path) as fd:
                lines = fd.readlines()
                lines = [line.strip() for line in lines]
            useful_paths = [path for path in useful_paths if lines[int(osp.split(path)[-1])] != '']
        if self.just_eval:
            res_paths = useful_paths
        else:
            if self.train:
                res_paths = useful_paths[:100]
            else:
                res_paths = useful_paths[100:]

        return res_paths

    def pc_loader(self, path):
        """
            Args:
                path:
            Returns:
                pc1: ndarray (N, 3) np.float32
                pc2: ndarray (N, 3) np.float32
        """
        pc1 = np.load(osp.join(path, 'pc1.npy'))  # .astype(np.float32)
        pc2 = np.load(osp.join(path, 'pc2.npy'))  # .astype(np.float32)
        return pc1, pc2

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
