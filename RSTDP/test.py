import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional, neuron, surrogate
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image
import numpy as np
from torchvision import transforms
from torch.optim import AdamW
from tqdm import tqdm
from spikingjelly.activation_based import functional
from train import Net, ANN, RstdpDataset

if __name__ == "__main__":
    data_path = r"classifier"
    model_path = r"model_stdp.pth"
    device = "cuda"
    dataset = RstdpDataset(path=data_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    rstdp_out = []
    rstdp_labels = []
    with torch.no_grad():
        total_sample = 0
        total_correct = 0
        for batch in tqdm(dataloader):
            functional.reset_net(model)
            img, label = batch[0], batch[1]
            img, label = img.to(device), label.to(device)
            output, mid_out = model(img)  #[B,6]
            rstdp_out.append(mid_out.cpu().numpy())
            rstdp_labels.append(label.cpu().numpy())
            total_sample += img.shape[0]
            total_correct += (torch.argmax(output, dim=1) == label).sum()
    rstdp_out = np.array(rstdp_out)
    rstdp_labels = np.array(rstdp_labels)
    np.save("input.npy", rstdp_out)
    np.save("labels.npy", rstdp_labels)
    print("Test Acc:{:4f}".format(total_correct / total_sample))
