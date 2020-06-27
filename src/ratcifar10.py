#!/usr/bin/env python3.7

from rat import rat
import shutil
import numpy as np

shutil.rmtree('logs', True)

if __name__ == '__main__':

    configs = {
        'classifier': [
            'vgg11_bn',
            'vgg13_bn',
            'vgg16_bn',
            'vgg19_bn',
            'resnet18',
            'resnet34',
            'resnet50',
            'densenet121',
            'densenet161',
            'densenet169',
            'mobilenet_v2',
            'googlenet',
            'inception_v3',
        ],
        'max_epochs': [100],
        'softmax_before_mean': [0],
        'num_students': [1],
        # 'max_epochs': [100],
        # 'num_students': [0],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'save_model': 1,
    }

    rat.run_experiment(configs, 'cifar10_train_ens.py', search_strategy={
        'create': 'sample',
        'queue_size': 3,
        'keep_best': -1,
        'score': 'summary_scalar',
        'args': {
            'score_key': 'val_acc',
            'lower_is_better': False,
            },
        }, step_after=False,
        )

    print('scheduled')
