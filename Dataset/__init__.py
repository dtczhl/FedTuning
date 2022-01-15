import os
import pathlib

DATASET_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Download')

# speech to command
DATASET_SPEECH_COMMAND_DIR = os.path.join(DATASET_DIR, 'speech_command/_FedTuning')

# emnist
DATASET_EMNIST_DIR = os.path.join(DATASET_DIR, 'emnist')

# emnist
DATASET_CIFAR100_DIR = os.path.join(DATASET_DIR, 'cifar100')
