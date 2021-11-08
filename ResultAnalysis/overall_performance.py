"""
    Overall performance for traces under Result/.
    Also, save processed data to Result/ProcessedData/
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

trace_infos = [
    # enable, dataset name, model name, initial M, initial E, alpha, beta, gamma, delta, penalty, trace id
    (False, 'speech_command', 'resnet_10', 20, 20, 0, 0, 0, 0, 1, 1),  # this is the baseline
    (True, 'speech_command', 'resnet_10', 20, 20, 1, 0, 0, 0, 1, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0, 0, 1, 0, 1, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0, 0, 1, 0, 5, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.25, 0.25, 0.25, 0.25, 1, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.25, 0.25, 0.25, 0.25, 10, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.1, 0, 0.1, 0.8, 1, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.1, 0, 0.1, 0.8, 10, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.1, 0, 0.1, 0.8, 10, 2),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.5, 0, 0, 0.5, 1, 1),
    (True, 'speech_command', 'resnet_10', 20, 20, 0.5, 0, 0, 0.5, 10, 2)
]

# --- End of Configuration ---



# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

# For ReadMe format
markdown_message = '| alpha | beta | gamma | delta | penalty | trace id | CompT (10^12) | TransT (10^6) | CompL (10^12) | TransL (10^6) | Final M | Final E | Overall |\n'
markdown_message += '| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n'

baseline_compT = baseline_transT = baseline_compL = baseline_transL = -1

for i_trace in range(len(trace_infos)):

    trace_info = trace_infos[i_trace]

    file_stat, filename = read_trace(trace_info)

    enable, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id = trace_info

    # round_id, model_accuracy, *eta_zeta_arr

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

    combined = np.vstack((round_id, model_accuracy, eta_t, eta_q, eta_z, eta_v, zeta_t, zeta_q, zeta_z, zeta_v, M, E,
                          compT, transT, compL, transL)).T
    df = pd.DataFrame(combined, columns=['round', 'accuracy', 'eta_t', 'eta_q', 'eta_z', 'eta_v',
                                         'zeta_t', 'zeta_q', 'zeta_z', 'zeta_v', 'M', 'E',
                                         'compT', 'transT', 'compL', 'transL'])
    df.to_csv(f'{project_dir}/Result/ProcessedData/{filename.split(".")[0]}.csv', index=False)

    if i_trace == 0:
        baseline_compT = compT_tot
        baseline_transT = transT_tot
        baseline_compL = compL_tot
        baseline_transL = transL_tot

    overall_improvement = alpha * (compT_tot - baseline_compT) / baseline_compT + \
                          beta * (transT_tot - baseline_transT) / baseline_compT + \
                          gamma * (compL_tot - baseline_compL) / baseline_compL + \
                          delta * (transL_tot - baseline_transL) / baseline_transL
    overall_improvement *= -100  # negative is improvement, so switch the sign

    print(f'dataset: {dataset_name}, model: {model_name} | '
          f'alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta: {delta}, penalty: {penalty}, trace_id: {trace_id} | '
          f'CompT (10^12): {compT_tot/10**12:.2f}, TransT (10^6): {transT_tot/10**6:.2f}, '
          f'CompL (10^12): {compL_tot/10**12:.2f}, TransL (10^6): {transL_tot/10**6:.2f} | '
          f'final M: {final_M}, final E: {final_E} | '
          f'overall: {np.format_float_positional(overall_improvement, precision=2, sign=True)}%')

    if i_trace == 0:
        markdown_message += f'| - | - | - | - | - | {trace_id} | ' \
                            f'{compT_tot/10**12:.2f} |  {transT_tot/10**6:.2f} | {compL_tot/10**12:.2f} | {transL_tot/10**6:.2f} | ' \
                            f'{int(final_M)} | {final_E} | - |\n'
    else:
        markdown_message += f'| {alpha} | {beta} | {gamma} | {delta} | {penalty} | {trace_id} | ' \
                            f'{compT_tot/10**12:.2f} |  {transT_tot/10**6:.2f} | {compL_tot/10**12:.2f} | {transL_tot/10**6:.2f} | ' \
                            f'{int(final_M)} | {final_E} | {np.format_float_positional(overall_improvement, precision=2, sign=True)}% |\n'

print('\n ------ Below for ReadMe ------\n')
print(markdown_message)

