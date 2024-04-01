import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models import get_model
from torchmetrics import CalibrationError
import random
import numpy as np

# Set the CUDA device based on the UUID

random_seed = 42
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

def evaluate(model_name, model_path, data_dir, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations
    transform_pipeline = transforms.Compose([
        ResizeAndCenterCrop((256, 256), 224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load CIFAR100 test data
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_pipeline)
    cal_dataset, val_dataset, test_dataset = random_split(test_dataset, [1000, 1000, len(test_dataset) - 2000])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(cal_dataset, batch_size=batch_size, shuffle=False)  # should be used for calibration

    # Initialize the model
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    ece = CalibrationError(task="multiclass", num_classes=100)
    correct = 0
    total = 0
    predictions = []  # List to store predictions' probabilities
    true_labels = []  # List to store true labels

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            ece.update(outputs.softmax(dim=-1), labels)

            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the {model_name} on the CIFAR100 test images: {accuracy}%')

    cal = ece.compute()
    print(f"ECE before scaling - {cal * 100:.3f}%.")
    print("saving true labels and uncalibrated probabilities..")
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    np.save('mobnetv3smalluncal_cifar100_predictions.npy', predictions)
    np.save('labels.npy', true_labels)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR100 model")
    parser.add_argument("--model_name", type=str, choices=['mobilenet_v3', 'efficientnet_v2_l'], required=True, help="Model to evaluate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weights")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for storing CIFAR100 data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")

    args = parser.parse_args()
    evaluate(args.model_name, args.model_path, args.data_dir, args.batch_size)
