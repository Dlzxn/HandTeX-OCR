import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets


class Data(Dataset):
    def __init__(self, path: str):
        """
        Инициализация класса с созданием кастомного трансформера
        :param path:
        """
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.RandomInvert(),
                                             transforms.RandomRotation(30),
                                             transforms.Resize((64, 64)),
                                             ])
        self.dataset = datasets.ImageFolder(path, transform=self.transform)

    def get_dataset(self):
        """
        Получение полноценного датасета
        :return: dataset
        """
        return self.dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
