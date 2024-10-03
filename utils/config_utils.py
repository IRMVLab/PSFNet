# -- coding: utf-8 --
import os
import sys

from importlib import import_module


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--config', type=str, default='config/cfg_psf_train.py', help='train config file path')
    args = parser.parse_args()
    config_path = os.path.abspath(os.path.expanduser(args.config))
    assert os.path.isfile(config_path)
    if config_path.endswith('.py'):
        module_name = os.path.basename(config_path)[:-3]
        config_dir = os.path.dirname(config_path)
        sys.path.insert(0, config_dir)
        mod = import_module(module_name)
        sys.path.pop(0)
        return mod.config
    else:
        raise IOError('Only .py type are supported now!')
