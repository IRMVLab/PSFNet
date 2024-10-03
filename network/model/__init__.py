# -- coding: utf-8 --
import copy

from .PSFModel import PSFModel


support_model = {'PSFModel': PSFModel}

__all__ = ['build_model']


def build_model(model_config):
    copy_config = copy.deepcopy(model_config)
    model_type = copy_config.pop('type')
    assert model_type in support_model, f'model_type {model_type} must in {support_model.keys()}'
    model = support_model[model_type](**copy_config)
    return model