"""
    plot accuracy vs penalty

    results copied from the output of performance_summary.py
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

from ResultAnalysis.ReadTrace import read_traces_of_preference_penalty

dataset_name = 'speech_command'
model_name = 'resnet_10'
initial_M = 20
initial_E = 20

preference_arr = [
    (0, 0.5, 0.5, 0),
    (0, 0.5, 0, 0.5),
    (0.33, 0.33, 0, 0.33)
]
trace_legend = [str(preference) for preference in preference_arr]
plt_fmt = ['o-', '^-', 's-']

penalty_factor_arr = np.arange(1, 11)
trace_id_arr = [1, 2, 3]

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

baseline = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=(0, 0, 0, 0), penalty=1, trace_id_arr=trace_id_arr)

# result_preference_matrix = np.zeros((len(preference_arr), len(penalty_factor_arr)))
result_preference_matrix = [[None for j in range(len(penalty_factor_arr))] for i in range(len(preference_arr))]
for i_preference, preference in enumerate(preference_arr):
    for i_penalty, penalty in enumerate(penalty_factor_arr):
        trace_result = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=preference, penalty=penalty, trace_id_arr=trace_id_arr, baseline=baseline)
        result_preference_matrix[i_preference][i_penalty] = trace_result

plt.figure(1, figsize=(7, 5))
for i in range(len(preference_arr)):
    plt.errorbar(penalty_factor_arr,
                 y=[100 * result.mean_improve_ratio for result in result_preference_matrix[i]],
                 yerr=[100 * result.std_improve_ratio for result in result_preference_matrix[i]],
                 fmt=plt_fmt[i], markersize=8, linewidth=3)
# plt.plot(penalty_factor_arr, [100 * result.mean_improve_ratio for result in result_preference_matrix[1]], '-^', linewidth=3)
# plt.plot(penalty_factor_arr, result_preference_matrix[2], '-*', linewidth=3)
plt.legend(trace_legend, loc='lower right', fontsize=18)
plt.xlabel('Penalty', fontsize=24)
plt.xlim([0.5, 10.5])
plt.xticks(penalty_factor_arr, fontsize=22)
plt.ylabel('Improvement Ratio (%)', fontsize=24)
plt.ylim([-80, 50])
# plt.yticks(np.arange(-60, 50, 20), fontsize=22)
plt.yticks(fontsize=22)
plt.grid(linestyle='--', linewidth=0.2)
plt.tight_layout()
image_filename = 'penalty_performance.jpg'
image_path = f'{project_dir}/Result/Image/{image_filename}'
print(f'saving image to {image_path}')
plt.savefig(image_path)
plt.show()



