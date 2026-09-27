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


class Net(nn.Module):

    def __init__(self, T=8, num_class=6):
        super(Net, self).__init__()
        self.T = T
        self.conv1 = nn.Sequential(layer.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=0, stride=2),
                                   neuron.IFNode(v_threshold=0.5, v_reset=None, surrogate_function=surrogate.ATan()))
        self.conv2 = nn.Sequential(layer.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride=2),
                                   neuron.IFNode(v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan()))
        self.flatten = layer.Flatten()  #[16,7,7]
        self.fc1 = nn.Sequential(layer.Linear(784, 16),
                                 neuron.IFNode(v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan()))
        self.fc2 = nn.Sequential(layer.Linear(16, num_class),
                                 neuron.IFNode(v_threshold=1., v_reset=None, surrogate_function=surrogate.ATan()))
        functional.set_step_mode(self, 'm')
        functional.set_backend(self, 'torch')

    def forward(self, x: torch.Tensor):
        #[B,C,H,W]
        x_T = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_T = self.conv1(x_T)
        x_T = self.conv2(x_T)
        x_T = self.flatten(x_T)
        x_T_mid = self.fc1(x_T)
        x_T = self.fc2(x_T_mid)
        return x_T.mean(0), x_T_mid


class ANN(nn.Module):

    def __init__(self, num_class=6):
        super(ANN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=0, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.relu2 = nn.ReLU()

        self.flatten = nn.Flatten()  #[16,7,7]
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, num_class)

    def forward(self, x: torch.Tensor):
        #[B,C,H,W]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x, x


class RstdpDataset(Dataset):

    def __init__(self, path, transform=None):
        super().__init__()
        self.path = path
        self.class_ids = [0, 5, 6, 7, 12, 19]
        self.transform = transform

        self.image_paths = []
        self.labels = []

        # 收集图像路径和标签
        for cls_id in self.class_ids:
            cls_folder = os.path.join(self.path, str(cls_id))
            for fname in os.listdir(cls_folder):
                self.image_paths.append(os.path.join(cls_folder, fname))
                self.labels.append(self.class_ids.index(cls_id))  # 映射为连续类 id
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = read_image(self.image_paths[index]).float() / 255.0  # C×H×W tensor
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    epochs = 50
    device = "cuda"
    transform = transforms.ConvertImageDtype(torch.float32)
    model_save_path = "model_stdp.pth"
    data_path = r"classifier"
    dataset = RstdpDataset(data_path, transform)
    model = Net()
    train_dataloder = DataLoader(dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    optimiezer = AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in tqdm(train_dataloder):
            functional.reset_net(model)
            img, label = batch[0], batch[1]
            img, label = img.to(device), label.to(device)
            output, _ = model(img)
            loss = criterion(output, label)
            total_loss += loss
            total_samples += img.shape[0]
            optimiezer.zero_grad()
            loss.backward()
            optimiezer.step()
        print("Epoch:{}  Loss:{:4f}".format(epoch, total_loss / total_samples))
    model.eval()
    with torch.no_grad():
        total_sample = 0
        total_correct = 0
        for batch in tqdm(test_dataloader):
            img, label = batch[0], batch[1]
            img, label = img.to(device), label.to(device)
            output, _ = model(img)  #[B,6]
            total_sample += img.shape[0]
            total_correct += (torch.argmax(output, dim=1) == label).sum()
    print("Test Acc:{:4f}".format(total_correct / total_sample))
    torch.save(model.state_dict(), model_save_path)
