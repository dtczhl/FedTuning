"""
    linear prediction accuracy
"""


import datetime
import os
import pathlib
import sys
from pathlib import Path
import copy
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt

from ResultAnalysis.ReadTrace import read_trace

# ------ Configurations ------

# enable, dataset name, model name, initial M, initial E, alpha, beta, gamma, delta, penalty, trace id
trace_info = (True, 'speech_command', 'resnet_10', 20, 20, 0, 0.5, 0, 0.5, 6, 1)


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


print(trace_matrix)
