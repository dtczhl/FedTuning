"""
    Download emnist dataset

    We use the LEAF Github for downloading and some preprocessing
    <https://github.com/TalwalkarLab/leaf>

    Download files are saved to Download/speech_command/
"""

import os
import pathlib

# dataset name
dataset_name = 'shakespeare'

# absolute path to Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)

if os.path.isdir(dataset_dir):
    # remove {dataset_dir} if exists
    os.system(f'rm -rf {dataset_dir}')

# create {dataset_name} directory
os.mkdir(dataset_dir)

# absolute path to Dataset/leaf
leaf_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Dataset/leaf')

# absolute path to Dataset/leaf/data/femnist
leaf_shakespeare_dir = os.path.join(leaf_dir, 'data/shakespeare')

# swith to {leaf_emnist_dir}
os.chdir(leaf_shakespeare_dir)

# download
os.system('./preprocess.sh -s niid -sf 1.0 -k 0 -t user --tf 0.8')

# move data for training and testing
print(f'mv {leaf_shakespeare_dir}/data/train/all_data_0_1_keep_0_train_8.json {dataset_dir}/')
os.system(f'mv {leaf_shakespeare_dir}/data/train/all_data_0_1_keep_0_train_8.json {dataset_dir}/')
print(f'mv {leaf_shakespeare_dir}/data/test/all_data_0_1_keep_0_test_8.json {dataset_dir}/')
os.system(f'mv {leaf_shakespeare_dir}/data/test/all_data_0_1_keep_0_test_8.json {dataset_dir}/')
#
# remove all other intermediate files
os.system(f'rm -rf {leaf_shakespeare_dir}/data/')
os.system(f'rm -rf {leaf_shakespeare_dir}/meta/')

print('Done!')

