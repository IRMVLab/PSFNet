# -- coding: utf-8 --
import copy

from addict import Dict
from torch.utils.data import DataLoader

from .kod import KOD
from .ksf142 import KSF142
from .lidarKITTI import lidarKITTI

support_dataset_set = {'KOD', 'KSF142', 'lidarKITTI'}
support_loader = {'DataLoader'}

def build_dataset(config):
    copy_config = copy.deepcopy(config)
    dataset_type = copy_config.pop('type')
    if not dataset_type in support_dataset_set:
        raise ValueError(f'{dataset_type} is not developed yet!, only {support_dataset_set} are support now')
    dataset = eval(dataset_type)(**copy_config)
    return dataset


def build_loader(dataset, config):
    dataloader_type = config.pop('type')
    if not dataloader_type in support_loader:
        raise ValueError(f'{dataloader_type} is not developed yet!, only {support_loader} are support now')

    # build collate_fn
    if 'collate_fn' in config:
        config['collate_fn']['dataset'] = dataset
        collate_fn = build_collate_fn(config.pop('collate_fn'))
    else:
        collate_fn = None
    dataloader = eval(dataloader_type)(dataset=dataset, collate_fn=collate_fn, **config ,pin_memory=True)
    return dataloader


def build_collate_fn(config):
    collate_fn_type = config.pop('type')
    if len(collate_fn_type) == 0:
        return None
    collate_fn_class = eval(collate_fn_type)(**config)
    return collate_fn_class

def build_dataloader(config):
    # build dataset
    copy_config = copy.deepcopy(config)
    copy_config = Dict(copy_config)
    dataset = build_dataset(copy_config.dataset)

    # build loader
    loader = build_loader(dataset, copy_config.loader)
    return loader
