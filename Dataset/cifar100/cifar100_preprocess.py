"""
    Pre-process Cifar100 dataset

    Preprocessed data are saved to Download/cifar100/
"""

import os
import pathlib
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from PIL import Image
from skimage.io import imsave


# dataset name
dataset_name = 'cifar100'

img_shape = (32, 32)

n_user_img = 50
np.random.seed(321)

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to /Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

train_file = os.path.join(dataset_dir, 'cifar-100-python/train')
test_file = os.path.join(dataset_dir, 'cifar-100-python/test')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


train_dataset = unpickle(train_file)
train_data = train_dataset[b'data']
train_label = train_dataset[b'fine_labels']

test_dataset = unpickle(test_file)
test_data = test_dataset[b'data']
test_label = test_dataset[b'fine_labels']

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
if os.path.isdir(train_dir):
    os.system(f'rm -rf {train_dir}')
os.system(f'mkdir {train_dir}')
if os.path.isdir(test_dir):
    os.system(f'rm -rf {test_dir}')
os.system(f'mkdir {test_dir}')

n_train_user = int(len(train_label) / n_user_img)
for i_user in range(n_train_user):
    user_data = train_data[i_user*n_user_img:(i_user+1)*n_user_img]
    user_label = train_label[i_user*n_user_img:(i_user+1)*n_user_img]

    user_dir = os.path.join(train_dir, str(i_user))
    assert not os.path.isdir(user_dir)
    os.makedirs(user_dir)

    for i_img in range(len(user_label)):

        img_data = user_data[i_img]
        img_label = user_label[i_img]

        r = np.array(img_data[:1024]).reshape(img_shape)
        g = np.array(img_data[1024:2048]).reshape(img_shape)
        b = np.array(img_data[2048:]).reshape(img_shape)
        img_jpg = np.dstack((r, g, b))

        img_filename = f'{i_img}_{img_label}.jpg'
        img_path = os.path.join(user_dir, img_filename)

        print(f' -> {img_path}')
        imsave(img_path, img_jpg)

n_test_user = int(len(test_label) / n_user_img)
for i_user in range(n_test_user):
    user_data = test_data[i_user*n_user_img:(i_user+1)*n_user_img]
    user_label = test_label[i_user*n_user_img:(i_user+1)*n_user_img]

    user_dir = os.path.join(test_dir, str(i_user))
    assert not os.path.isdir(user_dir)
    os.makedirs(user_dir)

    for i_img in range(len(user_label)):

        img_data = user_data[i_img]
        img_label = user_label[i_img]

        r = np.array(img_data[:1024]).reshape(img_shape)
        g = np.array(img_data[1024:2048]).reshape(img_shape)
        b = np.array(img_data[2048:]).reshape(img_shape)
        img_jpg = np.dstack((r, g, b))

        img_filename = f'{i_img}_{img_label}.jpg'
        img_path = os.path.join(user_dir, img_filename)

        print(f' -> {img_path}')
        imsave(img_path, img_jpg)


print(f'#train_user = {n_train_user}, #test_user = {n_test_user}')

os.system(f'rm -rf {os.path.join(dataset_dir, "cifar-100-python")}')

print('Done')


