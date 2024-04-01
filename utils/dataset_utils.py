import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random

random_seed = 42  # You can choose any integer
random.seed(random_seed)
torch.manual_seed(random_seed)

class ResizeAndCenterCrop:
    def __init__(self, resize_size, crop_size):
        self.resize = transforms.Resize(resize_size)
        self.center_crop = transforms.CenterCrop(crop_size)

    def __call__(self, img):
        img = self.resize(img)
        img = self.center_crop(img)
        return img

def get_cifar100_loaders(data_dir, batch_size):
    # Use a different name for the transform variable to avoid shadowing the transforms module
    transform_pipeline = transforms.Compose([
        ResizeAndCenterCrop((256, 256), 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # For mobnetv3 small
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_pipeline)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_pipeline)

    # Split the test dataset for calibration, validation, and test sets
    cal_dataset, val_dataset, remaining_test_dataset = random_split(test_dataset, [1000, 1000, len(test_dataset) - 2000])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(remaining_test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)  # For validation

    return train_loader, test_loader, val_loader
