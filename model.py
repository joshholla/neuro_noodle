import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms, utils, models

use_cuda = torch.cuda.is_available()

# ResNet 18!
# ------------------------------------------------------------------------------

def get_pretrained_model():
    model_ft = models.resnet18(pretrained=True)
    num_features = model_ft.fc.in_features
    # Changing up the last layer as we only have 2 classes:
    model_ft.fc = nn.Linear(num_features, 2) # nn.Linear(512,2)
    input_size = 224
    return model_ft, input_size
