"""
    Speech-to-command dataset for training

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


class SpeechCommandForTrain(Dataset):

    def __init__(self, user_dir):
        """ Each client's local data
        :param user_dir: directory for one client
        """

        self.user_dir = user_dir
        image_files = glob.glob(os.path.join(self.user_dir, '*.jpg'))
        self.X_filenames = [os.path.basename(image_file) for image_file in image_files]
        # label names
        self.y_label_names = [re.split('[_]', x)[0] for x in self.X_filenames]

        self.transform = transforms.Compose([
            transforms.Resize(SPEECH_COMMAND_INPUT_RESIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[SPEECH_COMMAND_TRAIN_MEAN],
                std=[SPEECH_COMMAND_TRAIN_STD]
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
        label_name = re.split('[_]', filename)[0]
        y = SPEECH_COMMAND_CLASSES.index(label_name)

        return X, y


if __name__ == '__main__':
    # For testing only
    print(f'--- Testing {__file__}')

    one_user_dir = f'{DATASET_DIR}/speech_command/_FedTuning/train/fffcabd1'
    print(f'\t--- One user dir: {one_user_dir}')

    one_user_data = SpeechCommandForTrain(user_dir=one_user_dir)

    print(f'\t\t--- self.X_filenames: {one_user_data.X_filenames}')
    print(f'\t\t--- self.y_label_names: {one_user_data.y_label_names}')

    print(f'--- User tensor data ---')
    for x in one_user_data:
        print(x)

