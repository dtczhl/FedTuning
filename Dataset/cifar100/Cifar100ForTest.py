"""
    Cifar100 dataset for testing

    Interface for DataLoader
"""

import glob
import re
import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from Dataset import DATASET_DIR
from Dataset.cifar100 import *


class Cifar100ForTest(Dataset):

    def __init__(self):
        """ For all the users in the test folder
        """

        self.test_image_files = []
        self.test_image_label_names = []

        test_dir = os.path.join(DATASET_DIR, 'cifar100/test')
        all_test_users = os.listdir(test_dir)
        for user_dir in all_test_users:
            image_files = glob.glob(os.path.join(test_dir, user_dir, '*.jpg'))
            image_label_names = [re.split('[_\.]', x)[1] for x in image_files]

            self.test_image_files.extend(image_files)
            self.test_image_label_names.extend(image_label_names)

        self.transform = transforms.Compose([
            transforms.Resize(CIFAR100_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CIFAR100_TRAIN_MEAN,
                std=CIFAR100_TRAIN_STD
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

        im = Image.open(filename).convert('RGB')
        X = self.transform(im)
        y = int(re.split('[_\.]', filename)[1])

        return X, y


if __name__ == '__main__':
    # For debugging purpose
    print(f'--- Debugging {__file__}')

    test_dataset = Cifar100ForTest()
    print(f'\t--- Total of samples for testing: {len(test_dataset)}')
