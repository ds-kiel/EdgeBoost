from torchmetrics import CalibrationError

from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.post_processing import TemperatureScaler
from utils.dataset_utils import get_cifar100_loaders
from models import get_model
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import os
import numpy as np



class ResizeAndCenterCrop:
    def __init__(self, resize_size, crop_size):
        self.resize = transforms.Resize(resize_size)
        self.center_crop = transforms.CenterCrop(crop_size)

    def __call__(self, img):
        img = self.resize(img)
        img = self.center_crop(img)
        return img


def calibrate(model_name,model_path,data_dir,batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   

    model = get_model(model_name)
    model=model.to(device)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    
    



    transform_pipeline = transforms.Compose([
        ResizeAndCenterCrop((256, 256), 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # For mobnetv3 small
    ])
    
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_pipeline)

    # Split the test dataset for calibration, validation, and test sets
    cal_dataset, val_dataset, remaining_test_dataset = random_split(test_dataset, [1000, 1000, len(test_dataset) - 2000])

    testloader = DataLoader(remaining_test_dataset, batch_size=batch_size, shuffle=False)

    ece = CalibrationError(task="multiclass", num_classes=100)
    my_device = torch.device("cpu")
    model.to(my_device)
    print("Model moved to CPU")


    scaler = TemperatureScaler()
    scaler = scaler.fit(model=model, calibration_set=cal_dataset)

    cal_model = torch.nn.Sequential(model, scaler)

    predictions = []  # List to store predictions' probabilities
    true_labels = []  # List to store true labels
    test_loss = 0.0
    correct = 0
    total = 0
# Reset the ECE
    ece.reset()

# Iterate on the test dataloader
    for inputs, labels in testloader:
        outputs = cal_model(inputs)
        ece.update(outputs.softmax(-1), labels)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        
        predictions.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Loss: {test_loss / len(testloader)}")
    print(f"Accuracy on test set: {100 * correct / total}%")

    cal = ece.compute()

    print(f"ECE after scaling - {cal*100:.3}%.")
    np.save('mobnetv3smallcal_cifar100_predictions.npy', predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate and evaluate a model on CIFAR100 dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
    parser.add_argument("--model_name", type=str, choices=['mobilenet_v3', 'efficientnet_v2_l'], required=True, help="Model name.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CIFAR100 dataset.")
    
    args = parser.parse_args()

    calibrate(args.model_name, args.model_path, args.data_dir)
import numpy as np
import matplotlib.pyplot as plt
