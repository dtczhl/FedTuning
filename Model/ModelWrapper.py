"""
    Wrapper class.

    Based on dataset and model structure, it selects the correct model for it.
"""

import re

from Model.ResNet import ResNet
from Dataset.speech_command import SPEECH_COMMAND_N_CLASS, SPEECH_COMMAND_N_INPUT_FEATURE


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
        else:
            print(f'unknown dataset_name {dataset_name}')
            exit(-1)

        model_detail = re.split('[_]', model_name)

        if model_detail[0] == 'resnet':
            n_layers = int(model_detail[1])
            return ResNet(depth=n_layers, num_input_feature=n_input_feature, num_classes=n_target_class)
        else:
            print(f'unknown model structure {model_detail[0]}')
            exit(-1)
