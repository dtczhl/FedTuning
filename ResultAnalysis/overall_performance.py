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

import numpy as np
import pandas as pd

from ResultAnalysis.ReadTrace import read_traces_of_preference_penalty

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

baseline = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=(0, 0, 0, 0), penalty=1, trace_id_arr=trace_id_arr)


for preference in preference_combine_all:

    for penalty in penalty_arr:
        trace_result = read_traces_of_preference_penalty(
            dataset_name=dataset_name, model_name=model_name, initial_M=initial_M, initial_E=initial_E,
            preference=preference, penalty=penalty, trace_id_arr=trace_id_arr, baseline=baseline)
        map_key = np.append(preference, [penalty])
        result_map[tuple(preference)] = trace_result

# For ReadMe format
markdown_message = '| alpha | beta | gamma | delta | penalty | trace id | CompT (10^12) | TransT (10^6) | CompL (10^12) | TransL (10^6) | Final M | Final E | Overall |\n'
markdown_message += '| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n'

markdown_message += f'| - | - | - | - | - | {trace_id_arr} | ' \
                    f'{baseline.mean_system[0] / 10 ** 12:.2f} ({baseline.std_system[0] / 10 ** 12:.2f}) | ' \
                    f'{baseline.mean_system[1] / 10 ** 6:.2f} ({baseline.std_system[1] / 10 ** 6:.2f}) | ' \
                    f'{baseline.mean_system[2] / 10 ** 12:.2f} ({baseline.std_system[2] / 10 ** 12:.2f}) | ' \
                    f'{baseline.mean_system[3] / 10 ** 6:.2f} ({baseline.std_system[3] / 10 ** 6:.2f}) | ' \
                    f'{baseline.mean_final_M:.2f} ({baseline.std_final_M:.2f}) | ' \
                    f'{baseline.mean_final_E:.2f} ({baseline.std_final_E:.2f}) | - |\n'

for preference in preference_combine_all:

    preference = tuple(preference)
    traces_result = result_map[preference]

    markdown_message += f'| {preference[0]} | {preference[1]} | {preference[2]} | {preference[3]} | {penalty} | {trace_id_arr} | ' \
                            f'{traces_result.mean_system[0]/10**12:.2f} ({traces_result.std_system[0]/10**12:.2f}) | ' \
                            f'{traces_result.mean_system[1]/10**6:.2f} ({traces_result.std_system[1]/10**6:.2f}) | ' \
                            f'{traces_result.mean_system[2]/10**12:.2f} ({traces_result.std_system[2]/10**12:.2f}) | ' \
                            f'{traces_result.mean_system[3]/10**6:.2f} ({traces_result.std_system[3]/10**6:.2f}) | ' \
                            f'{traces_result.mean_final_M:.2f} ({traces_result.std_final_M:.2f}) | ' \
                            f'{traces_result.mean_final_E:.2f} ({traces_result.std_final_E:.2f}) | ' \
                            f'{np.format_float_positional(100*traces_result.mean_improve_ratio, precision=2, sign=True)}% ' \
                            f'({100*traces_result.std_improve_ratio:.2f}%) |\n'


print('\n ------ Below for ReadMe ------\n')
print(markdown_message)

