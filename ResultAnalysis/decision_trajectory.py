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
import re
import math

import numpy as np
import matplotlib.pyplot as plt

from ResultAnalysis.ReadTrace import read_trace

# ------ Configurations ------

# enable, dataset name, model name, initial M, initial E, alpha, beta, gamma, delta, penalty, trace id
trace_info = (False, 'speech_command', 'resnet_10', 20, 20, 0, 0, 0, 0, 1, 1)


# --- End of Configuration ---

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

trace_matrix, filename = read_trace(trace_info=trace_info)
trace_matrix = np.array(trace_matrix)

M_trajectory = trace_matrix[:, 10]
E_trajectory = trace_matrix[:, 11]

alpha, beta, gamma, delta = trace_info[5:9]
penalty = trace_info[9]

plt.figure(1, figsize=(12, 10))
# plt.figure(1, figsize=(6, 5))
X_list = np.arange(len(M_trajectory))
plt.plot(X_list, M_trajectory, '-o', linewidth=3)
plt.plot(X_list, E_trajectory, '-^', linewidth=3)
plt.legend(['#participant', '#training pass'], loc='best', fontsize=30, ncol=2)
plt.xlim([0, max(X_list)])
# plt.xlabel('Training round', fontsize=24)
# plt.ylabel(fontsize=24)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.grid(linestyle='--', linewidth=1)
# plt.title(f'({alpha}, {beta}, {gamma}, {delta}), penalty={penalty}', fontsize=20)
plt.tight_layout()
image_filename = re.split('\.', filename)[0] + '.jpg'
image_path = f'{project_dir}/Result/Image/{image_filename}'
print(f'saving image to {image_path}')
plt.savefig(image_path)
plt.show()


