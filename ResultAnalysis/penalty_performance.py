"""
    plot accuracy vs penalty

    results copied from the output of performance_summary.py
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

improvement_ratio = [
    [10.99, 14.21, 13.58, 16.95, 0.47, 11.29, 4.13, 12.73, 11.44, 16.66],
    [6.36, -0.23, 21.79, -3.58, 14.4, 11.52, -1.21, 13.26, 19.30, -1.62],
    # [-57.73, 10.87, 19.22, 34.24, 28.50, 12.37, 18.18, 5.99, 16.29, 19.16]
]
trace_legend = [
    '(0.25, 0.25, 0.25, 0.25)',
    '(0, 0.33, 0.33, 0.33)',
    # '(0.33, 0, 0.33, 0.33)'
]
penalty_factor = np.arange(1, len(improvement_ratio[0])+1)

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

plt.figure(1, figsize=(6, 5))
plt.plot(penalty_factor, improvement_ratio[0], '-o', linewidth=3)
plt.plot(penalty_factor, improvement_ratio[1], '-^', linewidth=3)
# plt.plot(penalty_factor, improvement_ratio[2], '-*', linewidth=3)
plt.legend(trace_legend, loc='upper center', fontsize=20)
plt.xlabel('Penalty', fontsize=24)
plt.xlim([1, 10])
plt.xticks(penalty_factor, fontsize=22)
plt.ylabel('Improvement Ratio (%)', fontsize=24)
plt.ylim([-10, 40])
# plt.yticks(np.arange(-60, 50, 20), fontsize=22)
plt.yticks(fontsize=22)
plt.grid(linestyle='--', linewidth=0.2)
plt.tight_layout()
image_filename = 'penalty_performance.jpg'
image_path = f'{project_dir}/Result/Image/{image_filename}'
print(f'saving image to {image_path}')
plt.savefig(image_path)
plt.show()



