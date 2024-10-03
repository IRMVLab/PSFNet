# -- coding: utf-8 --
import os
import copy
import torch
import logging

from addict import Dict


class BaseNet(object):
    def __init__(self, model, train_config: Dict, logger: logging.Logger):
        super().__init__()
        self.net = model
        self.logger = logger
        self.devices = self.get_devices_ids(train_config.devices)

        self.logger.info(f'Number of {self.net.__class__.__name__} parameters: {self.get_para_num()}')
        self.optimizer = self.creat_optimizer(train_config.optimizer)
        self.scheduler = self.creat_scheduler(train_config.scheduler)
        self.global_state = {}

        self.load_model(train_config.resume_from)
        self.move_model(self.devices)

    def get_devices_ids(self, devices_config) -> list:
        example = f"devices is {devices_config}，should be ['cpu'] or ['cuda:0']"
        devices_config = list(devices_config)
        if (len(devices_config) == 0):
            self.logger.error(example)
            raise ValueError(example)
        if not 'cuda' in devices_config[0] and not 'cpu' in devices_config[0]:
            self.logger.error(example)
            raise ValueError(example)
        if not torch.cuda.is_available() or devices_config[0] == 'cpu':
            return [torch.device('cpu')]
        else:
            return [torch.device(x) for x in devices_config]

    def move_model(self, devices: list):
        if len(devices) > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=devices)
        self.net.to(devices[0])
        if not self.optimizer is None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(devices[0])
        self.logger.info(f'{self.net.__class__.__name__} trained on : {devices}')

    def load_model(self, ckpt_path: str):
        self.global_state = {}
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=self.devices[0])
            checkpoint_pre = self.strip_prefix(checkpoint['model_state_dict'])
            self.net.load_state_dict(checkpoint_pre)
            self.logger.info(f'{self.net.__class__.__name__} loads pretrain model {ckpt_path}!')
            if 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.logger.info(f'Optimizer loads pretrain from {ckpt_path}!')
            if 'scheduler' in checkpoint.keys():
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.logger.info(f'Scheduler loads pretrain from {ckpt_path}!')
            self.global_state = checkpoint.setdefault("global_state", dict())
        else:
            self.logger.info(f'train from scratch!')


    def creat_optimizer(self, config):
        if config:
            optimizer_config = copy.deepcopy(config)
            optimizer_type = optimizer_config.pop('type')
            optimizer = eval('torch.optim.{}'.format(optimizer_type))(self.net.parameters(), **optimizer_config)
            self.logger.info(f'the optimizer of your model {self.net.__class__.__name__} is {optimizer}!')
            return optimizer
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no optimizer!')
            return None

    def creat_scheduler(self, config):
        if config:
            scheduler_config = copy.deepcopy(config)
            scheduler_type = scheduler_config.pop('type')
            scheduler = eval('torch.optim.lr_scheduler.{}'.format(scheduler_type))(self.optimizer, **scheduler_config)
            self.logger.info(f'the scheduler of your model {self.net.__class__.__name__} is {scheduler.__class__.__name__}!')
            return scheduler
        else:
            self.logger.warning(f'your model {self.net.__class__.__name__} has no scheduler!')
            return None

    def get_para_num(self):
        return sum([p.data.nelement() for p in self.net.parameters()])

    def get_learing_rate(self) -> str:

        lr_str = 'param_group '
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr_str += f"{i}-th {param_group['lr']:.8f}"
        return lr_str

    def save_checkpoint(self, checkpoint_path, **kwargs):
        save_state = {
            'model_state_dict': self.net.module.state_dict() if hasattr(self.net, 'module') else self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_state.update(kwargs)
        torch.save(save_state, checkpoint_path)
        self.logger.info(f'Save {self.net.__class__.__name__} to {checkpoint_path}')

    def strip_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        if not all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            stripped_state_dict[key[len(prefix):]] = state_dict.pop(key)
        return stripped_state_dict

    def add_prefix(self, state_dict: dict, prefix: str = 'module.') -> dict:
        if all(key.startswith(prefix) for key in state_dict.keys()):
            return state_dict
        stripped_state_dict = {}
        for key in list(state_dict.keys()):
            key2 = prefix + key
            stripped_state_dict[key2] = state_dict.pop(key)
        return stripped_state_dict
