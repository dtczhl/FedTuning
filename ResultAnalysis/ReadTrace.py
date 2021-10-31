"""
    Read trace data
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import numpy as np

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

model_complexity = {
    # dataset name__model name: (flop, size)
    'speech_command__resnet_10': (12466403, 79715),
    'speech_command__resnet_18': (26794211, 177155),
    'speech_command__resnet_26': (41122019, 274595),
    'speech_command__resnet_34': (60119267, 515555)
}

def read_trace(trace_info):
    enable, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id = trace_info
    E_str = f'{initial_E:.2f}'.replace('.', '_')
    alpha_str = f'{alpha:.2f}'.replace('.', '_')
    beta_str = f'{beta:.2f}'.replace('.', '_')
    gamma_str = f'{gamma:.2f}'.replace('.', '_')
    delta_str = f'{delta:.2f}'.replace('.', '_')
    penalty_str = f'{penalty:.2f}'.replace('.', '_')
    filename = f'fedtuning_{enable}__{dataset_name}__{model_name}__M_{int(initial_M)}__E_{E_str}__' \
               f'alpha_{alpha_str}__beta_{beta_str}__gamma_{gamma_str}__delta_{delta_str}__penalty_{penalty_str}__{trace_id}.csv'

    ret = []

    with open(os.path.join(project_dir, 'Result', filename)) as f_in:
        while line_data := f_in.readline():

            line_fields = line_data.strip().split(',')
            round_id = int(line_fields[0])
            model_accuracy = float(line_fields[1])
            eta_zeta_arr = line_fields[2:10]
            M = int(line_fields[10])
            E = float(line_fields[11])
            cost_arr = [float(x) for x in line_fields[12:]]

            assert len(cost_arr) == M

            compT = max(cost_arr) * model_complexity[dataset_name + '__' + model_name][0]
            transT = 1.0 * model_complexity[dataset_name + '__' + model_name][1]
            compL = sum(cost_arr) * model_complexity[dataset_name + '__' + model_name][0]
            transL = len(cost_arr) * model_complexity[dataset_name + '__' + model_name][1]

            line_stat = [round_id, model_accuracy, *eta_zeta_arr, M, E, compT, transT, compL, transL]
            line_stat = [float(x) for x in line_stat]
            ret.append(line_stat)

    ret = np.array(ret)
    return ret


if __name__ == '__main__':

    info = (False, 'speech_command', 'resnet_10', 20, 20, 0, 0, 0, 0, 1)
    file_stat = read_trace(trace_info=info)
    print(file_stat)
