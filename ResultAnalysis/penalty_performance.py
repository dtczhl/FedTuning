"""
    plot accuracy vs penalty

    results copied from the output of performance_summary.py
"""

import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

improvement_ratio = [
    [4.91, 11.85, 9.19, 12.9, -3.92, 8.58, 2.44, 8.68, 8.4, 11.60],
    [13.05, -3.35, 25.81, -0.9, 19.75, 11.08, -8.34, 7.91, 20.64, 3.73],
    [-57.73, 10.87, 19.22, 34.24, 28.50, 12.37, 18.18, 5.99, 16.29, 19.16]
]
trace_legend = [
    '(0.25, 0.25, 0.25, 0.25)',
    '(0, 0.33, 0.33, 0.33)',
    '(0.33, 0, 0.33, 0.33)'
]
penalty_factor = np.arange(1, len(improvement_ratio[0])+1)

# temporarily add this project to system path
project_dir = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.append(project_dir)

plt.figure(1, figsize=(6, 5))
plt.plot(penalty_factor, improvement_ratio[0], '-o', linewidth=3)
plt.plot(penalty_factor, improvement_ratio[1], '-^', linewidth=3)
plt.plot(penalty_factor, improvement_ratio[2], '-*', linewidth=3)
plt.legend(trace_legend, loc='upper center', fontsize=20)
plt.xlabel('Penalty', fontsize=24)
plt.xlim([1, 10])
plt.xticks(penalty_factor, fontsize=22)
plt.ylabel('Improvement Ratio (%)', fontsize=24)
plt.ylim([-60, 120])
plt.yticks(np.arange(-60, 50, 20), fontsize=22)
plt.grid(linestyle='--', linewidth=0.2)
plt.tight_layout()
image_filename = 'penalty_performance.jpg'
image_path = f'{project_dir}/Result/Image/{image_filename}'
print(f'saving image to {image_path}')
plt.savefig(image_path)
plt.show()



