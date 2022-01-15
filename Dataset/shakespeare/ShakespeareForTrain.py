"""
    Shakespeare dataset for training

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


class ShakespeareForTrain(Dataset):

    def __init__(self, user_dir):
        """ Each client's local data
        :param user_dir: directory for one client
        """

        self.user_dir = user_dir

        x_file = os.path.join(self.user_dir, 'x.txt')
        y_file = os.path.join(self.user_dir, 'y.txt')

        with open(x_file) as f_x:
            self.X = f_x.readlines()
            self.X = [x[:-1] for x in self.X]
            self.X = [line.strip(r'\"') for line in self.X]

        with open(y_file) as f_y:
            self.y = f_y.readlines()
            self.y = [y[:-1] for y in self.y]


    def __len__(self):

        return len(self.X)

    def __getitem__(self, item):

        return self.X[item], self.y[item]


if __name__ == '__main__':

    # For debugging purpose
    print(f'--- Debugging {__file__}')

    one_user_dir = f'{DATASET_DIR}/shakespeare/train/TWELFTH_NIGHT__OR__WHAT_YOU_WILL_SECOND_OFFICER'
    print(f'\t--- One user dir: {one_user_dir}')

    one_user_data = ShakespeareForTrain(user_dir=one_user_dir)

    print(f'\t\t--- self.X: {one_user_data.X}')
    print(f'\t\t--- self.y: {one_user_data.y}')

    print(f'--- User tensor data ---')
    for x in one_user_data:
        print(x)

