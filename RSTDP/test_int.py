import numpy as np


def forward_int(data, weights, labels, threshold=1):
    pot = np.zeros([1, 5])
    weights = np.round(weights * 127)
    threshold = np.round(threshold * 127)
    spk_out = []
    for t in range(8):
        spk_T = data[t].reshape(-1)
        output = spk_T @ weights
        pot += output
        spk_out.append(pot > threshold)
        pot -= (pot > threshold) * threshold
    spk_out = np.array(spk_out).astype(np.int8)
    spk_out = spk_out.mean(0)
    return spk_out


def main():
    data = np.load('data_test.npy')
    labels = np.load('labels_test.npy')
    perm = np.random.permutation(len(data))
    data = data[perm]
    labels = labels[perm]
    num_classes = 5
    num_imgs = 20
    weights = np.load("weights.npy")
    correct = 0
    samples = 0
    for i in range(data.shape[0]):
        spk_input = data[i]
        label = labels[i]
        spk_out = forward_int(spk_input, weights, label)
        if label >= num_classes:
            break
        samples += 1
        if np.argmax(spk_out) == label:
            correct += 1
    print("Test Acc:{:4f}".format(correct / samples))


if __name__ == "__main__":
    main()
