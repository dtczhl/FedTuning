"""
    Number of right (negative) comparison function vs model accuracy
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

from ResultAnalysis.ReadTrace import read_traces_of_preference_penalty, read_trace


# ------ Configurations ------

dataset_name = 'speech_command'
model_name = 'resnet_10'
initial_M = 20
initial_E = 20
penalty_arr = [1, 10]
trace_id_arr = [1, 2, 3]

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
result_map_comparison = {}
result_map_comparison_base = {}

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

        comparison_result = []
        comparison_result_base = []
        # read raw traces
        for trace_id in trace_id_arr:
            trace_info = (True, dataset_name, model_name, initial_M, initial_E, *preference, penalty, trace_id)
            raw_trace, filename = read_trace(trace_info)

            alpha, beta, gamma, delta = preference

            round_id_arr = raw_trace[:, 0]
            model_accuracy_arr = raw_trace[:, 1]

            eta_zeta_arr_arr = raw_trace[:, 2:10]
            M_arr = raw_trace[:, 10]
            E_arr = raw_trace[:, 11]

            compT_arr = raw_trace[:, 12]
            transT_arr = raw_trace[:, 13]
            compL_arr = raw_trace[:, 14]
            transL_arr = raw_trace[:, 15]

            good_comparison = 0
            bad_comparison = 0

            a_prv = model_accuracy_arr[0]
            eps = 0.01
            compT_prv = compT_arr[0] / a_prv
            transT_prv = transT_arr[0] / a_prv
            compL_prv = compL_arr[0] / a_prv
            transL_prv = transL_arr[0] / a_prv

            for i_row in range(1, len(raw_trace)):

                a_cur = model_accuracy_arr[i_row]

                if a_cur - a_prv > eps:
                    compT_cur = compT_arr[i_row] / (a_cur - a_prv)
                    transT_cur = transT_arr[i_row] / (a_cur - a_prv)
                    compL_cur = compL_arr[i_row] / (a_cur - a_prv)
                    transL_cur = transL_arr[i_row] / (a_cur - a_prv)

                    I = alpha * (compT_cur - compT_prv) / compT_prv + \
                        beta * (transT_cur - transT_prv) / transT_prv + \
                        gamma * (compL_cur - compL_prv) / compL_prv + \
                        delta * (transL_cur - transL_prv) / transL_prv

                    if I <= 0:
                        good_comparison += 1
                    else:
                        bad_comparison += 1

                    compT_prv = compT_cur
                    transT_prv = transT_cur
                    compL_prv = compL_cur
                    transL_prv = transL_cur
                    a_prv = a_cur

            comparison_result.append((good_comparison, bad_comparison))

            # for baseline
            if penalty == 1:
                trace_info = (False, dataset_name, model_name, initial_M, initial_E, 0, 0, 0, 0, penalty, trace_id)
                raw_trace, filename = read_trace(trace_info)

                round_id_arr = raw_trace[:, 0]
                model_accuracy_arr = raw_trace[:, 1]
                eta_zeta_arr_arr = raw_trace[:, 2:10]
                M_arr = raw_trace[:, 10]
                E_arr = raw_trace[:, 11]
                compT_arr = raw_trace[:, 12]
                transT_arr = raw_trace[:, 13]
                compL_arr = raw_trace[:, 14]
                transL_arr = raw_trace[:, 15]

                good_comparison = 0
                bad_comparison = 0

                a_prv = model_accuracy_arr[0]
                eps = 0.01
                compT_prv = compT_arr[0] / a_prv
                transT_prv = transT_arr[0] / a_prv
                compL_prv = compL_arr[0] / a_prv
                transL_prv = transL_arr[0] / a_prv

                for i_row in range(1, len(raw_trace)):

                    a_cur = model_accuracy_arr[i_row]

                    if a_cur - a_prv > eps:
                        compT_cur = compT_arr[i_row] / (a_cur - a_prv)
                        transT_cur = transT_arr[i_row] / (a_cur - a_prv)
                        compL_cur = compL_arr[i_row] / (a_cur - a_prv)
                        transL_cur = transL_arr[i_row] / (a_cur - a_prv)

                        I = alpha * (compT_cur - compT_prv) / compT_prv + \
                            beta * (transT_cur - transT_prv) / transT_prv + \
                            gamma * (compL_cur - compL_prv) / compL_prv + \
                            delta * (transL_cur - transL_prv) / transL_prv

                        if I <= 0:
                            good_comparison += 1
                        else:
                            bad_comparison += 1

                        compT_prv = compT_cur
                        transT_prv = transT_cur
                        compL_prv = compL_cur
                        transL_prv = transL_cur
                        a_prv = a_cur

                comparison_result_base.append((good_comparison, bad_comparison))



        result_map_comparison[tuple(map_key)] = comparison_result

        if penalty == 1:
            result_map_comparison_base[tuple(map_key)] = comparison_result_base



result_compare_mean_comparison = np.zeros((len(preference_combine_all), 3))
result_compare_std_comparison = np.zeros((len(preference_combine_all), 3))
for i_preference, preference in enumerate(preference_combine_all):

    map_key_left = tuple(np.append(preference, [penalty_arr[0]]))
    map_key_right = tuple(np.append(preference, [penalty_arr[1]]))

    ratio_arr = []
    for good_bad in result_map_comparison[map_key_left]:
        # ratio = good_bad[0] / (good_bad[0] + good_bad[1])
        ratio = good_bad[1] / 100
        ratio_arr.append(ratio)
    result_compare_mean_comparison[i_preference, 0] = np.mean(ratio_arr)
    result_compare_std_comparison[i_preference, 0] = np.std(ratio_arr)

    ratio_arr = []
    for good_bad in result_map_comparison_base[map_key_left]:
        # ratio = good_bad[0] / (good_bad[0] + good_bad[1])
        ratio = good_bad[1] / 100
        ratio_arr.append(ratio)
    result_compare_mean_comparison[i_preference, 2] = np.mean(ratio_arr)
    result_compare_std_comparison[i_preference, 2] = np.std(ratio_arr)

    ratio_arr = []
    for good_bad in result_map_comparison[map_key_right]:
        # ratio = good_bad[0] / (good_bad[0] + good_bad[1])
        ratio = good_bad[1] / 100
        ratio_arr.append(ratio)
    result_compare_mean_comparison[i_preference, 1] = np.mean(ratio_arr)
    result_compare_std_comparison[i_preference, 1] = np.std(ratio_arr)


plt.figure(1, figsize=(12, 8))
x = np.arange(len(preference_combine_all))
x_tick_labels = []
for preference in preference_combine_all:
    x_tick_labels.append(str(preference))
bar_width = 0.40
plt.bar(x-bar_width/2, 100*result_compare_mean_comparison[:, 2], width=0.5*bar_width)
plt.bar(x, 100*result_compare_mean_comparison[:, 0], width=0.5*bar_width)
plt.bar(x+bar_width/2, 100*result_compare_mean_comparison[:, 1], width=0.5*bar_width)
# plt.ylabel('Good Ratio (%)', fontsize=24)
plt.ylabel('#Bad Decision', fontsize=24)
plt.xlabel('Training Preference', fontsize=24)
plt.yticks(fontsize=22)
plt.ylim([0, 30])
plt.xticks(x, labels=x_tick_labels, rotation=270, fontsize=18)
plt.legend(['Disabled', 'Penalty factor = 1', 'Penalty factor = 10'], loc='upper right', fontsize=20)
plt.errorbar(x-bar_width/2, 100*result_compare_mean_comparison[:, 2], yerr=100*result_compare_std_comparison[:, 2], capsize=5, ecolor='k', elinewidth=2, ls='none')
plt.errorbar(x, 100*result_compare_mean_comparison[:, 0], yerr=100*result_compare_std_comparison[:, 0], capsize=5, ecolor='k', elinewidth=2, ls='none')
plt.errorbar(x+bar_width/2, 100*result_compare_mean_comparison[:, 1], yerr=100*result_compare_std_comparison[:, 1], capsize=5, ecolor='k', elinewidth=2, ls='none')
plt.grid(linestyle='--', linewidth=0.2)
plt.tight_layout()
# plt.plot(x, 50*np.ones((len(x), 1)), linewidth=3)
image_filename = 'comparison_versus_performance.jpg'
image_path = f'{project_dir}/Result/Image/{image_filename}'
print(f'saving image to {image_path}')
plt.savefig(image_path)
plt.show()








