"""
    Speech-to-command dataset for testing

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


class SpeechCommandForTest(Dataset):

    def __init__(self):
        """ For all the users in the test folder
        """

        self.test_image_files = []
        self.test_image_label_names = []

        test_dir = os.path.join(DATASET_DIR, 'speech_command/_FedTuning/test')
        all_test_users = os.listdir(test_dir)
        for user_dir in all_test_users:
            image_files = glob.glob(os.path.join(test_dir, user_dir, '*.jpg'))
            image_label_names = [re.split('[_\/]', x)[-3] for x in image_files]

            self.test_image_files.extend(image_files)
            self.test_image_label_names.extend(image_label_names)

        self.transform = transforms.Compose([
            transforms.Resize(SPEECH_COMMAND_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[SPEECH_COMMAND_TRAIN_MEAN],
                std=[SPEECH_COMMAND_TRAIN_STD]
            )
        ])

    def __len__(self):
        """ Number of all testing set
        :return: number of testing samples
        """

        return len(self.test_image_files)

    def __getitem__(self, item):
        """ Get one sample based on index
        :param item: i.e., index
        :return: the sample of the index of item
        """

        filename = self.test_image_files[item]

        im = Image.open(filename).convert('L')
        X = self.transform(im)
        y = SPEECH_COMMAND_CLASSES.index(self.test_image_label_names[item])

        return X, y


if __name__ == '__main__':
    # For testing only
    print(f'--- Testing {__file__}')

    test_dataset = SpeechCommandForTest()
    print(f'\t--- Total of samples for testing: {len(test_dataset)}')

