import json
import os
import pathlib

# dataset name
dataset_name = 'shakespeare'

# absolute path to FedTuning/Download/
download_dir = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'Download')

# absolute path to /Download/{dataset_name}
dataset_dir = os.path.join(download_dir, dataset_name)
if not os.path.isdir(dataset_dir):
    print(f'Error: dataset directory {dataset_dir} does not exist')
    exit(-1)

train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')
if os.path.isdir(train_dir):
    os.system(f'rm -rf {train_dir}')
os.system(f'mkdir {train_dir}')
if os.path.isdir(test_dir):
    os.system(f'rm -rf {test_dir}')
os.system(f'mkdir {test_dir}')

train_file = os.path.join(dataset_dir, 'all_data_0_1_keep_0_train_8.json')
test_file = os.path.join(dataset_dir, 'all_data_0_1_keep_0_test_8.json')

with open(train_file) as f:
    user_data = json.load(f)

    users = user_data['users']
    for user in users:
        user_dir = os.path.join(train_dir, user)
        assert not os.path.isdir(user_dir)
        os.makedirs(user_dir)

        x_file = os.path.join(user_dir, 'x.txt')
        y_file = os.path.join(user_dir, 'y.txt')

        with open(x_file, "w") as f_x:
            for x in user_data['user_data'][user]['x']:
                f_x.write(x + '\n')

        with open(y_file, "w") as f_y:
            for y in user_data['user_data'][user]['y']:
                f_y.write(y + '\n')

with open(test_file) as f:
    user_data = json.load(f)

    users = user_data['users']
    for user in users:
        user_dir = os.path.join(train_dir, user)
        assert not os.path.isdir(user_dir)
        os.makedirs(user_dir)

        x_file = os.path.join(user_dir, 'x.txt')
        y_file = os.path.join(user_dir, 'y.txt')

        with open(x_file, "w") as f_x:
            for x in user_data['user_data'][user]['x']:
                f_x.write(x + '\n')

        with open(y_file, "w") as f_y:
            for y in user_data['user_data'][user]['y']:
                f_y.write(y + '\n')

os.system(f'rm {dataset_dir}/*.json')

print('Done')
