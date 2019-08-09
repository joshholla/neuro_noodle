"""
This file contains my NN models
"""
import torch
from torch import nn
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 channel_dim,
                 latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, channel_dim, 3, 1, 1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim, channel_dim, 3, 2, 1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim, latent_dim , 3, 1, 0),
            nn.BatchNorm2d(latent_dim)
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 channel_dim,
                 latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, channel_dim, 3, 1, 0),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_dim, channel_dim, 5, 1, 0),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_dim, channel_dim, 4, 2, 1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel_dim, output_dim, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.decoder(x)
        return out

class Autoencoder(torch.nn.Module):
    # The AutoEncoder Model
    def __init__(
            self,
            encoder,
            decoder,
            data_dim=506 * 650,
            channel_dim= 500 * 500,
            embed_dim=20
    ):
        super(Autoencoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

    def encode(self):
        # Encode a new image

    def decode(self):
        # Generate image from the decoder.

    def save_json(self):
        # Log the model Parameters
        # ------------------------------------------------------------------------------
        model_params = {
            # TODO - create json logs here!
        }
        json_params = json.dumps(model_params)
        text_file = open("model_params.json", "w")
        text_file.write(json_params)
        text_file.close()
        # TODO - send to comet.
        # if args.comet:
        #     args.experiment.log_metric("Blah", metric, step=time_step)
