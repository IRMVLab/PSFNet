# -- coding: utf-8 --

from addict import Dict

config = Dict()
config.exp_name = 'PSFNet'
config.SEED = 2022
config.model_option = {'type': "PSFModel", 'bn_decay': None}

MAX_EPOCH = 1000
config.train_option = {
    'devices': ['cuda:0'], 
    'resume_from': None,
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
    'train': {
        'kod': {
            'dataset': {
                'type':
                'KOD',
                'dataset_path':
                '/dataset/KITTI/data_odometry_velodyne/dataset',
                'seqs_list': [0, 1, 2, 3, 4, 5, 6],
                'check_seq_len':
                True,
                'pre_processes': [{
                    'type': 'RangeLimit',
                    'args': {
                        'x_range': [-30, 30],
                        'y_range': [-1, 1.4],
                        'z_range': [0, 35]
                    }
                }, {
                    'type': 'RandomSample',
                    'args': {
                        'num_points': 8192,
                        'allow_less_points': False
                    }
                }, {
                    'type': 'ShakeAug',
                    'args': {
                        'x_clip': 0.02,
                        'y_clip': 0.1,
                        'z_clip': 0.02
                    }
                }]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 8,
                'shuffle': True,
                'num_workers': 0,
            }
        }
    },
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
    },
    'pose_eval': {
        'seqs_list': [4], 
        'kod': {
            'dataset': {
                'type':
                'KOD',
                'dataset_path':
                '/dataset/KITTI/data_odometry_velodyne/dataset',
                'check_seq_len':
                True,
                'pre_processes': [{
                    'type': 'RangeLimit',
                    'args': {
                        'x_range': [-30, 30],
                        'y_range': [-1, 1.4],
                        'z_range': [0, 35]
                    }
                }, {
                    'type': 'RandomSample',
                    'args': {
                        'num_points': 8192,
                        'allow_less_points': False
                    }
                }]
            },
            'loader': {
                'type': 'DataLoader',
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 0,
            }
        }
    }
}

for k, v in config.items():
    if isinstance(v, dict):
        config[k] = Dict(v)

if __name__ == '__main__':
    pass
