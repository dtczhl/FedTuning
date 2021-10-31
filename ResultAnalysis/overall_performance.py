"""
    Overall performance
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import numpy as np

from ResultAnalysis.ReadTrace import read_trace

# ------ Configurations ------

trace_infos = [
    # enable, dataset name, model name, initial M, initial E, alpha, beta, gamma, delta, penalty, trace id
    (False, 'speech_command', 'resnet_10', 20, 20, 0, 0, 0, 0, 1, 1),  # this must be the baseline
    (True, 'speech_command', 'resnet_10', 20, 20, 0.1, 0, 0.1, 0.8, 1, 1),
]


# --- End of Configuration ---



# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)


baseline_compT = baseline_transT = baseline_compL = baseline_transL = -1

for i_trace in range(len(trace_infos)):

    trace_info = trace_infos[i_trace]

    file_stat = read_trace(trace_info)

    enable, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id = trace_info

    final_M = file_stat[-1, -6]
    final_E = file_stat[-1, -5]

    compT = sum(file_stat[:, -4])
    transT = sum(file_stat[:, -3])
    compL = sum(file_stat[:, -2])
    transL = sum(file_stat[:, -1])

    if i_trace == 0:
        baseline_compT = compT
        baseline_transT = transT
        baseline_compL = compL
        baseline_transL = transL

    overall_improvement = alpha * (compT - baseline_compT) / baseline_compT + \
                          beta * (transT - baseline_transT) / baseline_compT + \
                          gamma * (compL - baseline_compL) / baseline_compL + \
                          delta * (transL - baseline_transL) / baseline_transL
    overall_improvement *= -100  # negative is improvement, so switch the sign

    print(f'dataset: {dataset_name}, model: {model_name}, trace_id: {trace_id} | '
          f'alpha: {alpha}, beta: {beta}, gamma: {gamma}, delta: {delta} | '
          f'CompT (10^12): {compT/10**12:.2f}, TransT (10^6): {transT/10**6:.2f}, '
          f'CompL (10^12): {compL/10**12:.2f}, TransL (10^6): {transL/10**6:.2f} | '
          f'final M: {final_M}, final E: {final_E} | '
          f'overall: {np.format_float_positional(overall_improvement, precision=2, sign=True)}%')

