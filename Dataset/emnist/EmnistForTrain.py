"""
    EMNIST dataset for training

    Interface for DataLoader
"""

import glob
import re
import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from Dataset import DATASET_DIR
from Dataset.emnist import *


class EmnistForTrain(Dataset):

    def __init__(self, user_dir):
        """ Each client's local data
        :param user_dir: directory for one client
        """

        self.user_dir = user_dir
        image_files = glob.glob(os.path.join(self.user_dir, '*.jpg'))
        self.X_filenames = [os.path.basename(image_file) for image_file in image_files]
        # label names
        self.y_label_names = [re.split('[_\.]', x)[1] for x in self.X_filenames]

        self.transform = transforms.Compose([
            transforms.Resize(EMNIST_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[EMNIST_TRAIN_MEAN],
                std=[EMNIST_TRAIN_STD]
            )
        ])

    def __len__(self):
        """ Number of local data samples
        :return: number of local data samples
        """

        return len(self.X_filenames)

    def __getitem__(self, item):
        """ Get one sample based on index
        :param item: i.e., index
        :return: the sample of the index of item
        """

        filename = self.X_filenames[item]
        file_path = os.path.join(self.user_dir, filename)

        im = Image.open(file_path).convert('L')
        X = self.transform(im)
        y = int(re.split('[_\.]', filename)[1])

        return X, y


if __name__ == '__main__':
    # For debugging purpose
    print(f'--- Debugging {__file__}')

    one_user_dir = f'{DATASET_DIR}/emnist/train/647'
    print(f'\t--- One user dir: {one_user_dir}')

    one_user_data = EmnistForTrain(user_dir=one_user_dir)

    print(f'\t\t--- self.X_filenames: {one_user_data.X_filenames}')
    print(f'\t\t--- self.y_label_names: {one_user_data.y_label_names}')

    print(f'--- User tensor data ---')
    for x in one_user_data:
        print(x)