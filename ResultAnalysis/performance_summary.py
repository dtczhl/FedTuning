"""
    Performance summary
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

from ResultAnalysis.ReadTrace import read_trace

# ------ Configurations ------

dataset_name = 'speech_command'
model_name = 'resnet_10'
initial_M = 20
initial_E = 20
penalty = 1
trace_ids = [1]

# --- End of Configuration ---


class TraceResult:

    def __init__(self, compT, transT, compL, transL, final_M, final_E):
        self.CompT = [compT]
        self.TransT = [transT]
        self.CompL = [compL]
        self.TransL = [transL]
        self.final_M = [final_M]
        self.final_E = [final_E]

        self.mean_system = [np.average(self.CompT), np.average(self.TransT), np.average(self.CompL), np.average(self.TransL)]
        self.std_system = [np.std(self.CompT), np.std(self.TransT), np.std(self.CompL), np.std(self.TransL)]

        self.mean_final_M = np.average(self.final_M)
        self.std_final_M = np.std(self.final_M)
        self.mean_final_E = np.average(self.final_E)
        self.std_final_E = np.std(self.final_E)

    def __add__(self, other):
        self.CompT.extend(other.CompT)
        self.TransT.extend(other.TransT)
        self.CompL.extend(other.CompL)
        self.TransL.extend(other.TransL)

        self.mean = [np.average(self.CompT), np.average(self.TransT), np.average(self.CompL), np.average(self.TransL)]
        self.std = [np.std(self.CompT), np.std(self.TransT), np.std(self.CompL), np.std(self.TransL)]

        self.mean_final_M = np.average(self.final_M)
        self.std_final_M = np.std(self.final_M)
        self.mean_final_E = np.average(self.final_E)
        self.std_final_E = np.std(self.final_E)

    def __str__(self):
        return f'System Mean: {self.mean_system}\n' \
               f'System Std: {self.std_system}\n' \
               f'CompT: {self.CompT}\n' \
               f'TransT: {self.TransT}\n' \
               f'CompL: {self.CompL}\n' \
               f'TransL: {self.TransL}'


# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

preference_combine_all = [
    (0, 0, 0, 0),  # FedTuning false, baseline
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

for preference in preference_combine_all:
    enable = True if sum(preference) > 0 else False
    alpha, beta, gamma, delta = preference
    for trace_id in trace_ids:
        trace_info = (enable, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id)
        file_stat, filename = read_trace(trace_info)

        round_id = file_stat[:, 0]
        model_accuracy = file_stat[:, 1]
        eta_t = file_stat[:, 2]
        eta_q = file_stat[:, 3]
        eta_z = file_stat[:, 4]
        eta_v = file_stat[:, 5]
        zeta_t = file_stat[:, 6]
        zeta_q = file_stat[:, 7]
        zeta_z = file_stat[:, 8]
        zeta_v = file_stat[:, 9]
        M = file_stat[:, 10]
        E = file_stat[:, 11]
        compT = file_stat[:, 12]
        transT = file_stat[:, 13]
        compL = file_stat[:, 14]
        transL = file_stat[:, 15]

        final_M = M[-1]
        final_E = E[-1]

        compT_tot = sum(compT)
        transT_tot = sum(transT)
        compL_tot = sum(compL)
        transL_tot = sum(transL)

        trace_result = TraceResult(compT_tot, transT_tot, compL_tot, transL_tot, final_M, final_E)

        map_key = (alpha, beta, gamma, delta)
        if map_key in result_map:
            result_map[map_key] += trace_result
        else:
            result_map[map_key] = trace_result

# For ReadMe format
markdown_message = '| alpha | beta | gamma | delta | penalty | trace id | CompT (10^12) | TransT (10^6) | CompL (10^12) | TransL (10^6) | Final M | Final E | Overall |\n'
markdown_message += '| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n'


system_baseline = None
for preference in preference_combine_all:

    preference = tuple(preference)
    traces_result = result_map[preference]

    if tuple(preference) == (0, 0, 0, 0):
        system_baseline = traces_result
        markdown_message += f'| - | - | - | - | - | {trace_ids} | ' \
                            f'{traces_result.mean_system[0]/10**12:.2f} ({traces_result.std_system[0]:.2f}) | ' \
                            f'{traces_result.mean_system[1]/10**6:.2f} ({traces_result.std_system[1]:.2f}) | ' \
                            f'{traces_result.mean_system[2]/10**12:.2f} ({traces_result.std_system[2]:.2f}) | ' \
                            f'{traces_result.mean_system[3]/10**6:.2f} ({traces_result.std_system[3]:.2f}) | ' \
                            f'{traces_result.mean_final_M:.2f} ({traces_result.std_final_M:.2f}) | ' \
                            f'{traces_result.mean_final_E:.2f} ({traces_result.std_final_E:.2f}) | - |\n'
    else:

        overall_improvement = preference[0] * (traces_result.mean_system[0] - system_baseline.mean_system[0]) / system_baseline.mean_system[0] + \
                              preference[1] * (traces_result.mean_system[1] - system_baseline.mean_system[1]) / system_baseline.mean_system[1] + \
                              preference[2] * (traces_result.mean_system[2] - system_baseline.mean_system[2]) / system_baseline.mean_system[2] + \
                              preference[3] * (traces_result.mean_system[3] - system_baseline.mean_system[3]) / system_baseline.mean_system[3]
        overall_improvement *= -100  # negative is improvement, so switch the sign

        markdown_message += f'| {preference[0]} | {preference[1]} | {preference[2]} | {preference[3]} | {penalty} | {trace_ids} | ' \
                            f'{traces_result.mean_system[0] / 10 ** 12:.2f} ({traces_result.std_system[0]:.2f}) | ' \
                            f'{traces_result.mean_system[1] / 10 ** 6:.2f} ({traces_result.std_system[1]:.2f}) | ' \
                            f'{traces_result.mean_system[2] / 10 ** 12:.2f} ({traces_result.std_system[2]:.2f}) | ' \
                            f'{traces_result.mean_system[3] / 10 ** 6:.2f} ({traces_result.std_system[3]:.2f}) | ' \
                            f'{traces_result.mean_final_M:.2f} ({traces_result.std_final_M:.2f}) | ' \
                            f'{traces_result.mean_final_E:.2f} ({traces_result.std_final_E:.2f}) | ' \
                            f'{np.format_float_positional(overall_improvement, precision=2, sign=True)}% |\n'


print('\n ------ Below for ReadMe ------\n')
print(markdown_message)

