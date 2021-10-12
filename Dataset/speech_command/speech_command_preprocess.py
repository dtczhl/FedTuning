"""
    Pre-process speech-to-command dataset

    Preprocessed data are saved to FedTuning/Download/speech_command/_FedTuning/
"""

import os
import pathlib
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import librosa


# dataset name
dataset_name = 'speech_command'

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to FedTuning/Download/speech_command
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

# absolute path to store pre-processed data. FedTuning/Download/speech_command/_FedTuning
dataset_fedtuning_dir = os.path.join(dataset_dir, '_FedTuning')

# users for validation and testing, the remains are for training
validation_list_file = 'validation_list.txt'
testing_list_file = 'testing_list.txt'

# user ids for train, valid, and test
#   set() is better than dict{} in this case, but anyway, it works...
user_train_dict = {}
user_valid_dict = {}
user_test_dict = {}

# Users for validation
with open(os.path.join(dataset_dir, validation_list_file), 'r') as f_in:
    lines = f_in.readlines()
    for line in lines:
        user_id = re.split('[/_]', line)[-3]
        user_valid_dict[user_id] = user_valid_dict.get(user_id, 0) + 1

# Users for testing
with open(os.path.join(dataset_dir, testing_list_file), 'r') as f_in:
    lines = f_in.readlines()
    for line in lines:
        user_id = re.split('[/_]', line)[-3]
        user_test_dict[user_id] = user_valid_dict.get(user_id, 0) + 1

# Users for training
root_dir, data_dirs = next(os.walk(dataset_dir))[:2]
for data_dir in data_dirs:

    # ignore _background_noise_ folder, and _FedTuning folder
    if data_dir.startswith("_"):
        continue

    data_dir_path = os.path.join(root_dir, data_dir)
    wav_files = glob.glob(os.path.join(data_dir_path, '*.wav'))

    for wav_file in wav_files:
        user_id = re.split('[/_]', wav_file)[-3]
        if user_id in user_valid_dict or user_id in user_test_dict:
            continue
        else:
            user_train_dict[user_id] = user_train_dict.get(user_id, 0) + 1

# change working directory to {dataset_dir}
os.chdir(dataset_dir)
if not os.path.isdir(os.path.join(dataset_dir, '_FedTuning')):
    os.system('mkdir _FedTuning')

# change working directory to {dataset_fedtuning_dir}, i.e., {dataset_dir}/_FedTuning/
os.chdir(dataset_fedtuning_dir)
# delete all files in directory
os.system('rm -rf *')
# create folders for train, valid, and test
os.system('mkdir train test valid')

# Pre-processing happens here: convert wav to spectrogram and then image, save to corresponding train, valid, and test
root_dir, data_dirs = next(os.walk(dataset_dir))[:2]
for data_dir in data_dirs:

    # save spectrogram to image
    def spectrogram_image(y, sr, out, hop_length, n_mels):
        def scale_minmax(X, min=0.0, max=1.0):
            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max - min) + min
            return X_scaled

        # use log-melspectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                              n_fft=hop_length * 2, hop_length=hop_length)
        mels = np.log(mels + 1e-9)  # add small number to avoid log(0)

        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy

        # save as jpg
        skimage.io.imsave(out, img)

    # ignore _background_noise_ folder and _FedTuning
    if data_dir.startswith("_"):
        continue

    data_dir_path = os.path.join(root_dir, data_dir)
    wav_files = glob.glob(os.path.join(data_dir_path, "*.wav"))

    for wav_file in wav_files:
        wav_file_split = re.split('[/_\.]', wav_file)

        user_id = wav_file_split[-4]
        # e.g., bird_nohash_0.jpg
        image_out_filename = (wav_file_split[-5] + '_'
                              + wav_file_split[-3] + '_'
                              + wav_file_split[-2] + '.jpg')

        n_mels = 64
        n_time_steps = 63
        hop_length = 256

        y, sr = librosa.load(wav_file, sr=16000)
        y = np.concatenate((y[:n_time_steps * hop_length], [0] * (n_time_steps * hop_length - len(y))))

        # main workload
        if user_id in user_train_dict:
            user_dir = os.path.join(dataset_fedtuning_dir, 'train', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)
        elif user_id in user_valid_dict:
            user_dir = os.path.join(dataset_fedtuning_dir, 'valid', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)
        else:
            user_dir = os.path.join(dataset_fedtuning_dir, 'test', user_id)
            if not os.path.isdir(user_dir):
                os.system('mkdir {}'.format(user_dir))
            img_file = os.path.join(user_dir, image_out_filename)
            print(f'{wav_file} -> {img_file}')
            spectrogram_image(y, sr=sr, out=img_file, hop_length=hop_length, n_mels=n_mels)

print('Done!')
