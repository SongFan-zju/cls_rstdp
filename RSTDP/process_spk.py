import os
import numpy as np
import csv

num_classes = 5
num_imgs = 20

data = np.load("input.npy")
idx_list = np.load("index.npy").tolist()[:num_imgs]
data_test = np.load("data_test.npy")
# 创建主文件夹
base_folder = "test_spk"
os.makedirs(base_folder, exist_ok=True)

# 遍历 idx_list
for idx, spk in enumerate(data_test):
    spk_input = spk.astype(np.int8)
    output_file = os.path.join(base_folder, f"{idx}.txt")
    # 保存到 CSV 文件
    with open(output_file, "w", newline="") as f:
        for i in range(spk_input.shape[0]):
            spk = spk_input[i, :, :]
            i = np.where(spk > 0)[1] + 5
            hex_idx = np.vectorize(hex)(i)
            for j in hex_idx:
                # print("08088200000000" + j[2:].zfill(2))
                f.write("08088200000000" + j[2:].zfill(2) + "\n")
            f.write("0300000000001000\n")
            # TODO 这边少一个那个时间结束的标志
        f.write("0300000100001000\n")
    print(f"Saved {output_file}")

base_folder = "train_spk"
os.makedirs(base_folder, exist_ok=True)
for cnt, idx in enumerate(idx_list):
    spk = data[idx]
    spk_input = spk.astype(np.int8)
    output_file = os.path.join(base_folder, f"{idx}.txt")
    # 保存到 CSV 文件
    with open(output_file, "w", newline="") as f:
        for i in range(spk_input.shape[0]):
            spk = spk_input[i, :, :]
            i = np.where(spk > 0)[1] + 5
            hex_idx = np.vectorize(hex)(i)
            for j in hex_idx:
                # print("08088200000000" + j[2:].zfill(2))
                f.write("08088200000000" + j[2:].zfill(2) + "\n")
            f.write("0300000000001000\n")
            # TODO 这边少一个那个时间结束的标志
        f.write("0300000100001000\n")
    print(f"Saved {output_file}")
