import numpy as np
import os
from collections import defaultdict
import pickle as pkl
import random
import matplotlib.pyplot as plt


def forward_once(data, weights, label, threshold=127, rstdp=True):
    pot = np.zeros([1, 5])
    spk_out = []
    trace = np.zeros(16)
    delta_w = []
    for t in range(8):
        trace /= 2
        spk_T = data[t].reshape(-1)
        output = spk_T @ weights
        pot += output
        trace += spk_T / 2 * 16
        spk_out.append(pot > threshold)
        pot -= (pot > threshold) * threshold
        delta_w.append(trace[0].item())
    spk_out = np.array(spk_out).astype(np.int8)
    spk_out = spk_out.mean(0)
    output_label = np.argmax(spk_out)
    if rstdp == False:
        return spk_out, weights
    if output_label == label:
        delta_w = [x * 0.25 for x in delta_w]
    else:
        delta_w = [x * -0.5 for x in delta_w]
    return delta_w


if __name__ == "__main__":
    num_classes = 5
    weights = np.full([16, num_classes], fill_value=0.0)
    weights[:, 0] = 0.1
    weights = np.round(weights * 127)
    spk_input = np.zeros([8, 16])
    spk_input[0, :] = 1

    delta_w_plus = forward_once(spk_input, weights, threshold=30, label=0)[-5:-1]
    delta_w_mins = forward_once(spk_input, weights, threshold=30, label=1)[-5:-1]

    x = list(range(-4, 5))  # -8, -7, ..., 0, ..., 7, 8
    L = len(delta_w_mins)
    y_plus = [-x for x in delta_w_plus[::-1]] + [0] + delta_w_plus
    y_mins = [-x for x in delta_w_mins[::-1]] + [0] + delta_w_mins

    plt.figure(figsize=(8, 5))

    plt.scatter(x, y_plus, label='reward=0.25', color='blue', marker='o', s=60, zorder=3)
    plt.scatter(x, y_mins, label="reward=-0.5", color='red', marker='s', s=60, zorder=3)

    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    plt.axvline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)

    # 添加标签和标题
    plt.xlabel('T', fontsize=12)
    plt.ylabel('Δw', fontsize=12)

    # 添加图例和网格
    plt.legend(loc='best', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 设置 x 轴刻度
    plt.xticks(range(-L, L + 1, 2))

    # 设置图形边界
    plt.xlim(-0.5 - L, L + 0.5)
    plt.tight_layout()

    # 显示图形
    plt.savefig("rstdp.jpg")
    plt.show()
    print("reward=0.25:", y_plus)
    print("reward=-0.5", y_mins)
