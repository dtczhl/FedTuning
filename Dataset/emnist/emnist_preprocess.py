"""
    Pre-process EMNIST dataset

<<<<<<< HEAD
    Preprocessed data are saved to Download/emnist/
=======
    Preprocessed data are saved to Download/emnist/_FedTuning/
>>>>>>> origin/main
"""

import os
import pathlib
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
<<<<<<< HEAD
from PIL import Image
=======
>>>>>>> origin/main


# dataset name
dataset_name = 'emnist'

<<<<<<< HEAD
# split ratio of train vs test
split_train_test = 0.7
np.random.seed(321)

=======
>>>>>>> origin/main
# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to /Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

<<<<<<< HEAD
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
if os.path.isdir(train_dir):
    os.system(f'rm -rf {train_dir}')
os.system(f'mkdir {train_dir}')
if os.path.isdir(test_dir):
    os.system(f'rm -rf {test_dir}')
os.system(f'mkdir {test_dir}')

dataset_file = os.path.join(dataset_dir, 'emnist-byclass.mat')
dataset_mat = scipy.io.loadmat(dataset_file)

all_users = np.unique(dataset_mat['dataset'][0][0][0][0][0][2])
np.random.shuffle(all_users)

n_train_users = int(len(all_users) * split_train_test)
train_users = set(all_users[:n_train_users])
test_users = set(all_users[n_train_users:])
print(f'Total users: {len(all_users)}, where Train: {len(train_users)}, Test: {len(test_users)}')

n_img_tot = len(dataset_mat['dataset'][0][0][0][0][0][0])
for i_img in range(n_img_tot):
    img_user = dataset_mat['dataset'][0][0][0][0][0][2][i_img][0]
    img_dir = None
    if img_user in train_users:
        img_dir = os.path.join(train_dir, str(img_user))
    elif img_user in test_users:
        img_dir = os.path.join(test_dir, str(img_user))
    else:
        print(f'!!! Error, not a user')
        exit(-1)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    i_label = dataset_mat['dataset'][0][0][0][0][0][1][i_img][0]
    # filename: i_img_label
    img_filename = f'{i_img}_{i_label}.jpg'
    img_path = os.path.join(img_dir, img_filename)

    img_data = dataset_mat['dataset'][0][0][0][0][0][0][i_img]
    img_data = np.array(img_data)
    assert len(img_data) == 28*28
    img_data = np.reshape(img_data, (28, 28), order='F').astype(np.uint8)
    img_jpg = Image.fromarray(img_data)

    print(f'{i_img}/{n_img_tot} -> {img_path}')
    img_jpg.save(img_path)

# delete emnist-byclass.mat
os.system(f'rm {dataset_file}')

print('Done!')
=======
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
>>>>>>> origin/main
