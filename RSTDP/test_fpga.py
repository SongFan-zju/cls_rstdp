import run_pcss_fpga_rstdp as fpga
import numpy as np
import os
import cv2
from tqdm import tqdm
from rstdp import forward_once, test

port_id = 1
ip = os.environ.get("PCSS_FPGA_IP", "127.0.0.1")


def main():

    # #首先先没有训练先测试一下
    label_test = np.load("labels_test.npy")
    labels = np.load("labels.npy", allow_pickle=True)
    train_indexs = np.load("index.npy", allow_pickle=True).tolist()
    input_file = "img_path.txt"
    train_data_folder = "train_spk"
    test_data_folder = "test_spk"
    with open(input_file, "r") as f:
        img_path = [line.strip().replace("\\", "/") for line in f]
    # tmp = cv2.imread(img_path[0])
    # tmp = cv2.resize(tmp, (640, 640))
    # cv2.imshow("test", tmp)
    # cv2.waitKey(0)

    #遍历输入test值 fpga_spk/test文件下面，然后返回return_arr和return _arr.mean()
    #然后展示一下predict和true label和图片
    fpga.run_pcss(filename="flit_config/config_classify.txt", recv=False, ip=ip, port=port_id)
    files = os.listdir(test_data_folder)

    def test_fpga():
        acc = 0
        for cnt, path in tqdm(enumerate(files),
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
            test_label = label_test[cnt]
            test_flit_file = os.path.join(test_data_folder, f"{cnt}.txt")
            # print(test_flit_file, test_label)
            _, output_label = fpga.run_pcss(filename=test_flit_file, ip=ip, port=port_id)
            if test_label == output_label:
                acc += 1
            reward_flit = [
                "0e08820c00000000",
                "0e08820c00200000",
                "0e08820c00400000",
                "0e08820c00600000",
                "0e08820c00800000",
                "0300000000001000",
                "0300000100000100",
            ]
            fpga.run_pcss(flit_list=reward_flit, ip=ip, port=port_id)
            fpga.run_pcss(filename="flit_config/config_switch_input.txt", ip=ip, port=port_id)

        # print("Test Acc:{:4f}".format(acc / len(files)))
        return acc / len(files)

    print("FPGA原型系统")
    acc = test_fpga()
    print("Before Train H_Acc:{:4f}%".format(acc * 100))
    pbar = tqdm(total=19)
    for i, index in enumerate(train_indexs[:20]):
        imgpath = img_path[index]
        train_flit_file = os.path.join(train_data_folder, f"{index}.txt")
        true_label = labels[index]
        # print(train_flit_file, true_label)
        _, output_label = fpga.run_pcss(filename=train_flit_file, ip=ip, port=port_id)
        # print(f"output_label:{output_label}")
        reward_flit = [
            "0e08820c00000000",
            "0e08820c00200000",
            "0e08820c00400000",
            "0e08820c00600000",
            "0e08820c00800000",
            "0300000000001000",
            "0300000100000100",
        ]
        true_label = true_label.item()
        output_label = output_label.item()
        if true_label == output_label:
            # 替换最后两位为"10"，创建新字符串并替换列表元素
            reward_flit[true_label] = reward_flit[true_label][:-2] + "10"
        else:
            reward_flit[true_label] = reward_flit[true_label][:-2] + "10"
            reward_flit[output_label] = reward_flit[output_label][:-2] + "f0"
        # print("Reward Flit:", reward_flit)
        fpga.run_pcss(flit_list=reward_flit, ip=ip, port=port_id)
        fpga.run_pcss
        fpga.run_pcss(filename="flit_config/config_switch_input.txt", ip=ip, port=port_id)
        if i < 19:
            pbar.update(1)
        else:
            pbar.update(0)
        # 保存weight
        # weights = fpga.read_wgt(filename="flit_config/config_classify_read_wgt.txt", ip=ip, port=1234)
        # np.savetxt("fpga_weights.txt", weights, fmt="%d")
        # print(weights)
    pbar.close()
    h_acc = test_fpga()
    print("After Train H_Acc:{:4f}%".format(h_acc * 100))

    print("算法软件：")
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
    s_acc = test(data_test, weights, labels_test)

    print("Before Train S_Acc:{:4f}%".format(acc * 100))
    for idx in idx_list:
        label = labels[idx][0]
        spk_input = data[idx]
        spk_out, weights = forward_once(spk_input, weights, label, rstdp=True)
        # spk_out_new, _ = forward_once(spk_input, weights, label, rstdp=False)
    # rstdp_acc = test(data, weights, labels)
    rstdp_acc = test(data_test, weights, labels_test)
    print("After {} imgs Train\nTest S_Acc:{:4f}%".format(num_imgs - 1, rstdp_acc * 100))


if __name__ == "__main__":
    main()
