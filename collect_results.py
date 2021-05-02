import argparse
import collections
import json
import os
import numpy as np
import dataset
import mlconfig
import models
import torch
import util
from evaluator import Evaluator
from tabulate import tabulate

parser = argparse.ArgumentParser(description='ClasswiseNoise')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    print("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')
print("PyTorch Version: %s" % (torch.__version__))


def load_results(targt_exp, model_name):
    # print(targt_exp)
    config_file = os.path.join(targt_exp, model_name+'.yaml')
    checkpoint_path_file = os.path.join(targt_exp, 'checkpoints', model_name)
    if not os.path.isfile(config_file) or not os.path.isfile(checkpoint_path_file+'.pth'):
        # print('No such files: \n%s\n%s' % (config_file, checkpoint_path_file))
        return None

    config = mlconfig.load(config_file)
    config.set_immutable()
    model = config.model().to(device)
    checkpoints = util.load_model(filename=checkpoint_path_file, model=model, optimizer=None, scheduler=None)
    if config.epochs != checkpoints['epoch']:
        return None
    if 'cm_history' in checkpoints['ENV']:
        new_hist = []
        for item in checkpoints['ENV']['cm_history']:
            if isinstance(item, np.ndarray):
                new_hist.append(item.tolist())
            else:
                new_hist.append(item)
        checkpoints['ENV']['cm_history'] = new_hist
    return checkpoints['ENV']


if __name__ == '__main__':
    exp_names = [
        'experiments/cifar10/random_samplewise/CIFAR10-eps=8',
        'experiments/cifar10/min-max_samplewise/CIFAR10-eps=8-se=0.9-base_version=resnet18',
        'experiments/cifar10/min-min_samplewise/CIFAR10-eps=8-se=0.1-base_version=resnet18',
        'experiments/cifar10/min-min_samplewise/CIFAR10-eps=8-se=0.01-base_version=resnet18',
        'experiments/cifar100/min-min_samplewise/CIFAR100-eps=8-se=0.3-base_version=resnet18',
        'experiments/cifar100/min-min_samplewise/CIFAR100-eps=8-se=0.01-base_version=resnet18',
        'experiments/svhn/min-min_samplewise/SVHN-eps=8-se=0.1-base_version=resnet18',
        'experiments/svhn/min-min_samplewise/SVHN-eps=8-se=0.01-base_version=resnet18',
        'experiments/imagenet-mini/min-min_samplewise/ImageNetMini-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar10/random_classwise/CIFAR10-eps=8/',
        'experiments/cifar10/min-max_classwise/CIFAR10-eps=8-se=0.8-base_version=resnet18',
        'experiments/cifar10/min-min_classwise/CIFAR10-eps=8-se=0.1-base_version=resnet18',
        'experiments/cifar10/min-min_classwise/CIFAR10-eps=8-se=0.01-base_version=resnet18',
        'experiments/cifar100/min-min_classwise/CIFAR100-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar100/min-min_classwise/CIFAR100-eps=8-se=0.01-base_version=resnet18',
        'experiments/svhn/min-min_classwise/SVHN-eps=8-se=0.1-base_version=resnet18',
        'experiments/svhn/min-min_classwise/SVHN-eps=8-se=0.01-base_version=resnet18',
        'experiments/imagenet-mini/min-min_classwise/ImageNetMini-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=24-se=0.01-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=24-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=24-se=0.01-base_version=resnet18',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=16-se=0.01-base_version=resnet18',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=24-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=24-se=0.01-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=8-se=0.1-base_version=resnet18-2noise',
        'experiments/cifar10-extension/min-min_classwise/TinyImageNet-eps=16-se=0.1-base_version=resnet18',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random8',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random16',
        'experiments/cifar10-extension/min-min_classwise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random24',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random8',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random16',
        'experiments/cifar10-extension/min-min_samplewise/CIFAR10-eps=8-se=0.1-base_version=resnet18-random24',
    ]

    model_list = [
        'resnet18',
        'resnet50',
        'dense121',
        'resnet18_augmentation',
        'resnet18_madrys',
        'resnet18_classpoison',
        'resnet18_classpoison_targeted',
        'resnet18_add-uniform-noise',
        'resnet18_add-uniform-noise-aug',
        'resnet18_cutout',
        'resnet18_cutmix',
        'resnet18_mixup',
    ]

    poison_rate_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    exp_results = {}
    for exp_name in exp_names:
        print(exp_name)
        table_data_header = ['Model'] + poison_rate_list
        table_data = [model_list]
        exp_results[exp_name] = {}
        for poison_rate in poison_rate_list:
            target_dir = os.path.join(exp_name, 'poison_train_%.1f' % poison_rate)
            temp_list = []
            exp_results[exp_name][poison_rate] = {}
            for model_name in model_list:
                rs_env = load_results(os.path.join(target_dir, model_name), model_name)
                exp_results[exp_name][poison_rate][model_name] = rs_env
                if rs_env is not None:
                    temp_list.append('%.2f' % rs_env['curren_acc'])
                else:
                    temp_list.append('..')
            table_data.append(temp_list)

        # Transpose array
        table_data = list(map(list, zip(*table_data)))

        print('=' * 40 + 'Results' + '=' * 40)
        print(tabulate(table_data, headers=table_data_header, floatfmt=".2f", stralign="left", numalign="left"))
        print('=' * (80 + len('Results')) + '\n')

    # Save results to
    with open('exp_results.json', 'w') as outfile:
        json.dump(exp_results, outfile)
