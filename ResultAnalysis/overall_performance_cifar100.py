"""
    Overall performance. Factor 10 vs 1
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ResultAnalysis.ReadTrace import read_traces_of_preference_penalty

# ------ Configurations ------

dataset_name = 'cifar100'
model_name = 'resnet_10'
initial_M = 20
initial_E = 20
penalty_arr = [10]
trace_id_arr = [1, 2]

# --- End of Configuration ---


# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

preference_combine_all = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 1),
    (1, 0, 1, 1),
    (0, 1, 1, 1),
    (1, 1, 1, 1)
]
preference_combine_all = np.array(preference_combine_all).astype(float)
for i_row in range(len(preference_combine_all)):
    row_sum = sum(preference_combine_all[i_row])
    if row_sum > 0:
        preference_combine_all[i_row] /= row_sum
float_2_decimal = lambda x: float('{:.2f}'.format(x))
vfunc = np.vectorize(float_2_decimal)
preference_combine_all = vfunc(preference_combine_all)

result_map = {}

baseline = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=(0, 0, 0, 0), penalty=1, trace_id_arr=trace_id_arr)


for preference in preference_combine_all:

    for penalty in penalty_arr:
        trace_result = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=preference, penalty=penalty, trace_id_arr=trace_id_arr, baseline=baseline)
        map_key = np.append(preference, [penalty])
        result_map[tuple(map_key)] = trace_result

result_compare_mean = np.zeros((len(preference_combine_all), 2))
result_compare_std = np.zeros((len(preference_combine_all), 2))
for i_preference, preference in enumerate(preference_combine_all):
    map_key_left = tuple(np.append(preference, [penalty_arr[0]]))
    # map_key_right = tuple(np.append(preference, [penalty_arr[1]]))
    result_compare_mean[i_preference, 0] = result_map[map_key_left].mean_improve_ratio
    # result_compare_mean[i_preference, 1] = result_map[map_key_right].mean_improve_ratio
    result_compare_std[i_preference, 0] = result_map[map_key_left].std_improve_ratio
    # result_compare_std[i_preference, 1] = result_map[map_key_right].std_improve_ratio


# plt.figure(1, figsize=(12, 8))
# x = np.arange(len(preference_combine_all))
# x_tick_labels = []
# for preference in preference_combine_all:
#     x_tick_labels.append(str(preference))
# bar_width = 0.40
# plt.bar(x-bar_width/2, 100*result_compare_mean[:, 0], width=0.9*bar_width)
# plt.bar(x+bar_width/2, 100*result_compare_mean[:, 1], width=0.9*bar_width)
# plt.ylabel('Improvement Ratio (%)', fontsize=24)
# plt.xlabel('Training Preference', fontsize=24)
# plt.yticks(fontsize=22)
# plt.xticks(x, labels=x_tick_labels, rotation=270, fontsize=18)
# plt.legend(['No penalty', 'Penalty factor = 10'], loc='lower left', fontsize=20)
# plt.errorbar(x-bar_width/2, 100*result_compare_mean[:, 0], yerr=100*result_compare_std[:, 0], capsize=5, ecolor='k', elinewidth=2, ls='none')
# plt.errorbar(x+bar_width/2, 100*result_compare_mean[:, 1], yerr=100*result_compare_std[:, 1], capsize=5, ecolor='k', elinewidth=2, ls='none')
# plt.grid(linestyle='--', linewidth=0.2)
# plt.tight_layout()
# image_filename = 'overall_performance.jpg'
# image_path = f'{project_dir}/Result/Image/{image_filename}'
# print(f'saving image to {image_path}')
# plt.savefig(image_path)
# plt.show()

print(f'Mean (std): '
      f'{100*np.mean(result_compare_mean[:, 0]):.2f}% ({100*np.mean(result_compare_std[:, 0]):.2f}%) '
      f'vs. {100*np.mean(result_compare_mean[:, 1]):.2f}% ({100*np.mean(result_compare_std[:, 1]):.2f}%)')


