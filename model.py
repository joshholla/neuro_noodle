import torch
from torch import nn
from torch.nn import functional as F


use_cuda = torch.cuda.is_available()


# ----------------------------------------------------------------------------------
#                         BUILDING OUR NEURAL NETWORK
# ----------------------------------------------------------------------------------
class Autoencoder(torch.nn.Module):
    # The AutoEncoder Model
    # ------------------------------------------------------------------------------
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(506 * 650, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 40), # check if it is 256, or something else
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 506 * 650),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    # For usage after training is done
    # ------------------------------------------------------------------------------

    def encode(self, x):
        # Generate latent representation of the input image
        x = self.encoder(x)
        return x

    def decode(self, x):
        # Generate image from the decoder.
        x = self.decoder(x)
        return x


def get_model():
    # Returns the model and optimizer
    model = Autoencoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer
