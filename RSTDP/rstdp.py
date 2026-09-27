import numpy as np
import os
from collections import defaultdict
import pickle as pkl
import random


def forward_once(data, weights, label, threshold=127, rstdp=True):
    pot = np.zeros([1, 5])
    spk_out = []
    trace = np.zeros(16)
    for t in range(8):
        trace /= 2
        spk_T = data[t].reshape(-1)
        output = spk_T @ weights
        pot += output
        trace += spk_T / 2
        spk_out.append(pot > threshold)
        pot -= (pot > threshold) * threshold

    spk_out = np.array(spk_out).astype(np.int8)
    spk_out = spk_out.mean(0)
    output_label = np.argmax(spk_out)
    if rstdp == False:
        return spk_out, weights
    if output_label == label:
        weights[:, output_label] += (trace / 8 * 127).astype(np.int8)
    else:
        weights[:, output_label] -= (trace / 8 * 127).astype(np.int8)
        weights[:, label] += (trace / 8 * 127).astype(np.int8)
    return spk_out, weights


def test(data, weights, labels):
    correct = 0
    samples = 0
    num_classes = 5
    corr_list = []
    for i in range(data.shape[0]):
        spk_input = data[i]
        label = labels[i]

        spk_out, _ = forward_once(spk_input, weights, label, rstdp=False)

        if label >= num_classes:
            break
        samples += 1
        if np.argmax(spk_out) == label:
            correct += 1
            corr_list.append(i)
    return correct / samples


if __name__ == "__main__":
    data = np.load("input.npy")
    data_test = np.load('data_test.npy')
    labels_test = np.load('labels_test.npy')
    labels = np.load("labels.npy", allow_pickle=True)
    num_classes = 5
    num_imgs = 20

    weights = np.full([16, num_classes], fill_value=0.1)
    weights = np.round(weights * 127)
    num_imgs = 20
    idx_list = np.load("index.npy").tolist()[:num_imgs]
    print("FPGA原型系统:")
    acc = test(data_test, weights, labels_test)

    print("Before Train Acc:{:4f}".format(acc))
    for idx in idx_list:
        label = labels[idx][0]
        spk_input = data[idx]
        spk_out, weights = forward_once(spk_input, weights, label, rstdp=True)
        # spk_out_new, _ = forward_once(spk_input, weights, label, rstdp=False)
    # rstdp_acc = test(data, weights, labels)
    rstdp_acc = test(data_test, weights, labels_test)
    print("After {} imgs Train\nTest S_Acc:{:4f}".format(num_imgs, rstdp_acc))
    print("Classify算法:")
    print("Before Train Acc:{:4f}".format(acc))
    print("After {} imgs Train\nTest H_Acc:{:4f}".format(num_imgs, rstdp_acc))

    axon = []
    for i in range(num_classes):
        tmp = []
        for j in range(num_classes, num_classes + weights.shape[0]):
            tmp.append([j, i])
        axon.append(tmp)
    # print(axon)
    all_weights = np.zeros([num_classes + weights.shape[0], num_classes + weights.shape[0]])
    all_weights[num_classes:, :num_classes] = weights
    # print(all_weights)
