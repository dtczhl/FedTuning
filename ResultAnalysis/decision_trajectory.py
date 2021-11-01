"""
    Visualize decision trajectories of M and E
"""

import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ------ Configurations ------

trace_infos = [
    # enable, dataset name, model name, initial M, initial E, alpha, beta, gamma, delta, penalty, trace id
    (True, 'speech_command', 'resnet_10', 20, 20, 0.1, 0, 0.1, 0.8, 1, 1),  # 1
]

# --- End of Configuration ---

model_complexity = {
    # dataset name__model name: (flop, size)
    'speech_command__resnet_10': (12466403, 79715),
    'speech_command__resnet_18': (26794211, 177155),
    'speech_command__resnet_26': (41122019, 274595),
    'speech_command__resnet_34': (60119267, 515555)
}


# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

for trace_info in trace_infos:
    enable, dataset_name, model_name, initial_M, initial_E, alpha, beta, gamma, delta, penalty, trace_id = trace_info
    E_str = f'{initial_E:.2f}'.replace('.', '_')
    alpha_str = f'{alpha:.2f}'.replace('.', '_')
    beta_str = f'{beta:.2f}'.replace('.', '_')
    gamma_str = f'{gamma:.2f}'.replace('.', '_')
    delta_str = f'{delta:.2f}'.replace('.', '_')
    penalty_str = f'{penalty:.2f}'.replace('.', '_')
    filename = f'fedtuning_{enable}__{dataset_name}__{model_name}__M_{int(initial_M)}__E_{E_str}__' \
               f'alpha_{alpha_str}__beta_{beta_str}__gamma_{gamma_str}__delta_{delta_str}__penalty_{penalty_str}__{trace_id}.csv'
    # read file data

    compT = 0
    transT = 0
    compL = 0
    transL = 0

    M_trajectory = []
    E_trajectory = []

    with open(os.path.join(project_dir, 'Result', filename)) as f_in:
        while line_data := f_in.readline():

            line_fields = line_data.strip().split(',')
            round_id = int(line_fields[0])
            model_accuracy = float(line_fields[1])
            eta_zeta_arr = line_fields[2:10]
            M = int(line_fields[10])
            E = float(line_fields[11])
            cost_arr = [float(x) for x in line_fields[12:]]

            M_trajectory.append(M)
            E_trajectory.append(E)

            assert len(cost_arr) == M

            compT += max(cost_arr)
            transT += 1.0
            compL += sum(cost_arr)
            transL += len(cost_arr)

    plt.figure(1)
    X_list = np.arange(len(M_trajectory))
    plt.plot(X_list, M_trajectory)
    plt.plot(X_list, E_trajectory)
    plt.legend(['M', 'E'])
    plt.xlabel('Training round')
    plt.show()

