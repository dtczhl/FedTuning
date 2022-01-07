"""
    Pre-process EMNIST dataset

    Preprocessed data are saved to Download/emnist/_FedTuning/
"""

import os
import pathlib
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.io


# dataset name
dataset_name = 'emnist'

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to /Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

dataset_file = os.path.join(dataset_dir, 'emnist-byclass.mat')
dataset_mat = scipy.io.loadmat(dataset_file)

# images
print(dataset_mat['dataset'][0][0][0][0][0][0].shape)
print(len(np.unique(dataset_mat['dataset'][0][0][0][0][0][1])))
print(len(np.unique(dataset_mat['dataset'][0][0][0][0][0][2])))

img1 = dataset_mat['dataset'][0][0][0][0][0][0][10]
img1 = np.array(img1)
img1 = np.reshape(img1, (28, 28), order='F')
plt.imshow(img1, cmap='gray')
plt.show()

print(dataset_mat['dataset'][0][0][0][0][0][1][10])
