"""
    Download emnist dataset.

    Downloaded files are saved to Download/emnist/
"""

import os
import pathlib

# dataset name
dataset_name = 'emnist'

# url to download the speech-to-command dataset
dataset_url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip'
# compressed file
compressed_filename = 'matlab.zip'

# absolute path to Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to Download/{dataset_name}
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
# move
os.system(f'mv matlab.zip {dataset_name}')
os.chdir(f'{dataset_name}')
os.system('pwd')
# uncompress dataset
os.system(f'unzip {compressed_filename}')
# remove compressed files to save a little bit disk
os.system(f'rm {compressed_filename}')
os.system(f'mv ./matlab/emnist-byclass.mat .')
os.system('rm -rf matlab')

print('Done!')
