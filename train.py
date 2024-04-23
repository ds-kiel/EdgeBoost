# EdgeOffload/train.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from utils.dataset_utils import get_cifar100_loaders
from models import get_model

def train(model_name, data_dir, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lr=0.01
    batch_size=32
    
    # Load data
    train_loader, test_loader, val_loader = get_cifar100_loaders(data_dir, batch_size)
    
    # Initialize model
    model = get_model(model_name).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Early stopping setup
    early_stopping_patience = 20
    best_val_loss = float('inf')
    current_patience = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_accuracy = accuracy_score(val_targets, val_predictions)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy * 100}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= early_stopping_patience:
                print("Early stopping triggered. Training finished.")
                break

    print("Finished Training")
    # Save the trained model
    torch.save(model.state_dict(), f"{model_name}_cifar100.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on CIFAR100 with early stopping and validation")
    parser.add_argument("--model_name", type=str, choices=['mobilenet_v3', 'efficientnet_v2_l'], required=True, help="Model to train")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for storing CIFAR100 data")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")

    args = parser.parse_args()
    train(args.model_name, args.data_dir, args.epochs)
