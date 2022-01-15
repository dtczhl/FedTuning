"""
    Wrapper class.

    Based on dataset and model structure, it selects the correct model for it.
"""

import re

from Model.ResNet import ResNet
from Model.VGG import VGG
from Model.EmnistCR import EmnistCR
from Model.LogisticRegression import LogisticRegression

from Dataset.speech_command import SPEECH_COMMAND_N_CLASS, SPEECH_COMMAND_N_INPUT_FEATURE
from Dataset.emnist import EMNIST_N_CLASS, EMNIST_N_INPUT_FEATURE
from Dataset.cifar100 import CIFAR100_N_CLASS, CIFAR100_N_INPUT_FEATURE


class ModelWrapper:

    @staticmethod
    def build_model_for_dataset(model_name, dataset_name):

        model_name = model_name.strip().lower()
        dataset_name = dataset_name.strip().lower()

        n_target_class = -1
        n_input_feature = -1
        if dataset_name == 'speech_command':
            n_target_class = SPEECH_COMMAND_N_CLASS
            n_input_feature = SPEECH_COMMAND_N_INPUT_FEATURE
        elif dataset_name == 'emnist':
            n_target_class = EMNIST_N_CLASS
            n_input_feature = EMNIST_N_INPUT_FEATURE
        elif dataset_name == 'cifar100':
            n_target_class = CIFAR100_N_CLASS
            n_input_feature = CIFAR100_N_INPUT_FEATURE
        else:
            print(f'unknown dataset_name {dataset_name}')
            exit(-1)

        model_detail = re.split('[_]', model_name)

        if model_detail[0] == 'resnet':
            n_layers = int(model_detail[1])
            return ResNet(depth=n_layers, num_input_feature=n_input_feature, num_classes=n_target_class)
        elif model_detail[0] == 'vgg':
            n_layers = int(model_detail[1])
            return VGG(depth=n_layers, num_input_feature=n_input_feature, num_classes=n_target_class)
        elif model_detail[0] == 'emnistcr':
            return EmnistCR(depth=-1, num_input_feature=n_input_feature, num_classes=n_target_class)
        elif model_detail[0] == 'logisticregression':
            return LogisticRegression(depth=-1, num_input_feature=n_input_feature, num_classes=n_target_class)
        else:
            print(f'unknown model structure {model_detail[0]}')
            exit(-1)
