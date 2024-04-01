# EdgeOffload/models/efficientnet_v2_l.py

from torchvision import models
import torch.nn as nn

def get_efficientnet_v2_l():
    model = models.efficientnet_v2_l(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1280, out_features=1000),
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=1000, out_features=100)
    )
    return model
