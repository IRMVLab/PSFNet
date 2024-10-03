# -- coding: utf-8 --
import numpy as np

from dataset.base_preprocess import BaseRangeLimit, BaseRandomSample, BaseTransform


class RangeLimitWithSF(BaseRangeLimit):
    def __init__(self, *, x_range=(-30, 30), y_range=(-1, 1.4), z_range=(0, 35), use_same_indice=False):

        super().__init__(x_range=x_range, y_range=y_range, z_range=z_range)
        self.use_same_indice = use_same_indice

    def __call__(self, data_dict):
        pc1 = data_dict['pc1']
        pc2 = data_dict['pc2']
        sf = data_dict['flow']
        indice1, indice2 = list(map(self.get_limit_range_indices, [pc1, pc2]))
        if self.use_same_indice:
            indice = list(set(indice1) & set(indice2))
            indice1 = np.array(indice)
            indice2 = np.array(indice)

        data_dict['pc1'] = pc1[indice1]
        data_dict['pc2'] = pc2[indice2]
        data_dict['flow'] = sf[indice1]
        return data_dict


class RandomSampleWithSF(BaseRandomSample):
    def __init__(self, *, num_points=8192, allow_less_points=False, no_corr=True):
        super().__init__(num_points, allow_less_points)
        self.no_corr = no_corr

    def __call__(self, data_dict):
        pc1 = data_dict['pc1']
        pc2 = data_dict['pc2']
        sf = data_dict['flow']
        if self.no_corr:
            indice1, indice2 = list(map(self.get_random_sample_indices, [pc1.shape[0], pc2.shape[0]]))
        else:
            indice1 = self.get_random_sample_indices(min(pc1.shape[0], pc2.shape[0]))
            indice2 = indice1
        data_dict['pc1'] = pc1[indice1]
        data_dict['pc2'] = pc2[indice2]
        data_dict['flow'] = sf[indice1]
        return data_dict


class TransformerWithSF(BaseTransform):
    def __init__(self, *, Tr=None):
        super().__init__()
        self.Tr = Tr

    def __call__(self, data_dict):
        data_dict['pc1'] = self.transformer_tr(data_dict['pc1'], self.Tr)
        data_dict['pc2'] = self.transformer_tr(data_dict['pc2'], self.Tr)
        data_dict['flow'] = self.transformer_tr(data_dict['flow'], self.Tr)
        return data_dict
