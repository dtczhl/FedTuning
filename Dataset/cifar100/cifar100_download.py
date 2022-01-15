"""
    Download Cifar100 dataset.

    Downloaded files are saved to FedTuning/Download/speech_command/
"""

import os
import pathlib

# dataset name
dataset_name = 'cifar100'

# url to download the speech-to-command dataset
dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
# compressed file
compressed_filename = 'cifar-100-python.tar.gz'

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to FedTuning/Download/speech_command
dataset_dir = os.path.join(download_dir, dataset_name)

# switch to FedTuning/Download/
os.chdir(download_dir)

if os.path.isdir(dataset_dir):
    # remove {dataset_dir} if exists
    os.system(f'rm -rf {dataset_dir}')

# create {dataset_name} directory
os.mkdir(dataset_name)
# download dataset
os.system(f'wget {dataset_url}')
# uncompress dataset
os.system(f'tar -xvf {compressed_filename} -C {dataset_name}')
# remove compressed files to save a little bit disk
os.system(f'rm {compressed_filename}')

print('Done!')
