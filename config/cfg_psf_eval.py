# -- coding: utf-8 --

from addict import Dict

config = Dict()
config.exp_name = 'PSFNet'
config.SEED = 2022
config.model_option = {'type': "PSFModel", 'bn_decay': None}

MAX_EPOCH = 1000
config.train_option = {
    'devices': ['cuda:0'],
    'resume_from': "please input the path of the checkpoint file",
    'optimizer': {
        'type': 'Adam',
        'lr': 0.001,
        'weight_decay': 1e-4
    },
    'scheduler': {
        'type': 'StepLR',
        'step_size': 13,
        'gamma': 0.7,
        'last_epoch': -1
    },
    'learning_rate_clip': 1e-5,
    'epochs': MAX_EPOCH,
    'print_interval': 50,
    'val_interval': 2,

    'train_schedule': [(0, 400), (400, 600), (600, MAX_EPOCH)],
    'output_save_dir': f"./output/{config.exp_name}",
    'ckpt_save_type': 'FixedEpochStep',
    'ckpt_save_epoch': 2,
}

config.dataset_option = {
    'sf_eval': {
        'ksf142': {
            'dataset': {
                'type':
                    'KSF142',
                'dataset_path':
                    '/dataset/sf_eval/ksf142',
                'just_eval':
                    True,
                'train':
                    False,
                'pre_processes': [{
                    'type': 'RangeLimitWithSF',
                    'args': {
                        'x_range': [-30, 30],
                        'y_range': [-1.4, 1],
                        'z_range': [0, 35]
                    },
                    'use_same_indice': True
                }, {
                    'type': 'RandomSampleWithSF',
                    'args': {
                        'num_points': 8192,
                        'allow_less_points': False
                    },
                    'no_corr': True
                }, {
                    'type': 'TransformerWithSF',
                    'args': {
                        'Tr': [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
                    }
                }]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            },
            'eval_2d': True
        },
        'lidarKITTI': {
            'dataset': {
                'type':
                    'lidarKITTI',
                'dataset_path':
                    '/dataset/sf_eval/lidarKITTI',
                'just_eval':
                    True,
                'train':
                    False,
                'pre_processes': [{
                    'type': 'RangeLimitWithSF',
                    'args': {
                        'x_range': [-30, 30],
                        'y_range': [-1.4, 1],
                        'z_range': [0, 35]
                    }
                }, {
                    'type': 'RandomSampleWithSF',
                    'args': {
                        'num_points': 8192,
                        'allow_less_points': False,
                        'no_corr': True
                    }
                }, {
                    'type': 'TransformerWithSF',
                    'args': {
                        'Tr': [[-1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
                    }
                }]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            },
            'eval_2d': False
        }
    }
}

for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)

if __name__ == '__main__':
    pass
