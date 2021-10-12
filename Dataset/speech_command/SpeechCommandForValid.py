"""
    Speech-to-command dataset for validation

    Interface for DataLoader
"""

import glob
import re
import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from Dataset import DATASET_DIR
from Dataset.speech_command import *


class SpeechCommandForValid(Dataset):

    def __init__(self):
        """ For all the users in the valid folder
        """

        self.valid_image_files = []
        self.valid_image_label_names = []

        valid_dir = os.path.join(DATASET_DIR, 'speech_command/_FedTuning/valid')
        all_valid_users = os.listdir(valid_dir)
        for user_dir in all_valid_users:
            image_files = glob.glob(os.path.join(valid_dir, user_dir, '*.jpg'))
            image_label_names = [re.split('[_\/]', x)[-3] for x in image_files]

            self.valid_image_files.extend(image_files)
            self.valid_image_label_names.extend(image_label_names)

        self.transform = transforms.Compose([
            transforms.Resize(SPEECH_COMMAND_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[SPEECH_COMMAND_TRAIN_MEAN],
                std=[SPEECH_COMMAND_TRAIN_STD]
            )
        ])

    def __len__(self):
        """ Number of all validation set
        :return: number of validation samples
        """

        return len(self.valid_image_files)

    def __getitem__(self, item):
        """ Get one sample based on index
        :param item: i.e., index
        :return: the sample of the index of item
        """

        filename = self.valid_image_files[item]

        im = Image.open(filename).convert('L')
        X = self.transform(im)
        y = SPEECH_COMMAND_CLASSES.index(self.valid_image_label_names[item])

        return X, y


if __name__ == '__main__':
    # For testing only
    print(f'--- Testing {__file__}')

    valid_dataset = SpeechCommandForValid()
    print(f'\t--- Total of samples for validation: {len(valid_dataset)}')

