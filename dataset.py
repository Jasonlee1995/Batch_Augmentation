import torch, torchvision
from torch.utils.data import Dataset


class BA_CIFAR10(Dataset):
    def __init__(self, M, cifar_dir, train, transform=None):
        self.dataset = torchvision.datasets.CIFAR10(cifar_dir, train=train, download=False, transform=transform)
        self.M = M

    def __getitem__(self, idx):
        imgs, labels = [], []
        for _ in range(self.M):
            img, label = self.dataset[idx]
            imgs.append(img)
            labels.append(label)
            
        imgs, labels = torch.stack(imgs), torch.LongTensor(labels)
        return imgs, labels

    def __len__(self):
        return len(self.dataset)