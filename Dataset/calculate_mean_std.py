

import glob
import re
import os

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from Dataset import DATASET_DIR
from Dataset.emnist import *


target_dir = 'cifar100/test'

user_dir = os.path.join(DATASET_DIR, target_dir)
all_users = os.listdir(user_dir)

n = 0
x = 0
x_1 = 0
delta2 = 0
delta2_1 = 0
for i_user, user in enumerate(all_users):
    print(f'{i_user} / {len(all_users)}')
    image_files = glob.glob(os.path.join(user_dir, user, '*.jpg'))
    for image_file in image_files:
        # im = np.array(Image.open(image_file).convert('L')) / 255.0  # for grayscale
        im = np.array(Image.open(image_file).convert('RGB'))[:, :, 2] / 255.0   # for rgb
        for row in im:
            for pixel in row:
                n += 1
                x = x_1 + (pixel - x_1) / n
                delta2 = delta2_1 + ((pixel - x_1) * (pixel - x) - delta2_1) / n

                x_1 = x
                delta2_1 = delta2

print(f'Mean = {x}, delta = {np.sqrt(delta2)}')

