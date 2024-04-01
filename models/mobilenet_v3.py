from torchvision import models
import torch.nn as nn

def get_mobilenet_v3_small():
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier = nn.Sequential()
    model.classifier.add_module('dropl', nn.Dropout(p=0.4, inplace=True))
    model.classifier.add_module('linr1', nn.Linear(in_features=576, out_features=1000))
    model.classifier.add_module('drop2', nn.Dropout(p=0.4, inplace=True))
    model.classifier.add_module('linr2', nn.Linear(in_features=1000, out_features=100))

    return model