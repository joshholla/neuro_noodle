import torch
from torch import nn
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()

class Autoencoder(torch.nn.Module):
    # The AutoEncoder Model
    def __init__(self,encoder=Encoder(),decoder=Decoder(),
                 data_dim=506 * 650,
                 #channel_dim= 500 * 500,
                 num_channels = 500,
                 embed_dim=20):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, 3, 2, 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, latent_dim , 3, 1, 0),
            nn.BatchNorm2d(latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_channels , 3, 1, 0),
            nn.BatchNorm2d( num_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels, num_channels, 5, 1, 0),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels, num_channels , 4, 2, 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_channels, output_dim, 4, 2, 1),
            nn.Tanh()
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

    # # TODO - this might be redundant.
    # def save_model(self):
    #     # Log the model Parameters
    #     # ------------------------------------------------------------------------------
    #     model_params = {
    #         # TODO - create json logs here!
    #     }
    #     json_params = json.dumps(model_params)
    #     text_file = open("model_params.json", "w")
    #     text_file.write(json_params)
    #     text_file.close()
