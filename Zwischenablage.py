import math
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


batch_size = 128
num_workers = 4

data_dir = Path("hands-on/data")
data_dir.mkdir(parents=True, exist_ok=True)


cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)
normalize = transforms.Normalize(mean=cifar_mean, std=cifar_std)

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transforms)
test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)

pin_memory = device == "cpu"
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

class_names = train_ds.classes
print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
