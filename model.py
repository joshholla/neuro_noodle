import torch
from torch import nn
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()

class Autoencoder(torch.nn.Module):
    # The AutoEncoder Model
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,1,3),
            nn.ReLU(inplace=True),
            nn.Linear(506 * 650, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 25, 3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.Conv2d(25, 50, 3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 1, 3),
            nn.BatchNorm2d(50),
            nn.Linear(256, 40), # check if it is 256, or something else
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.ConvTranspose2d(1, 25, 3),
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(25, 50, 3),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 1 , 3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Linear(256, 506 * 650),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        # Generate image from the decoder.
        x = self.decoder(x)
        return x
