import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import pdb
import argparse
import time

from model import *
from utils import *

def get_model():
    model = Autoencoder()
    return model, optim.Adam(model.parameters(), lr=0.001)


EPOCHS = 50
BATCH_SIZE = 32


def loss_batch(model, loss_function, image, opt=None):
    loss = loss_function(model(image), image) # we want to get the images
    # reconstructed when we come back out of the autoencoder.

    # so that we don't train on validation data (don't pass optimizer for
    # validation set:
    if opt is not None:
        loss.backward()
        opt.step
        opt.zero_grad()

    return loss.item() # What am I returning here?


# training loop
# ------------------------------------------------------------------------------

def fit (model, data, epochs=EPOCHS, batch_size= BATCH_SIZE, training_data, validation_data, optim):
    loss = torch.nn.BCELoss()
    for epoch in tqdm(range(epochs)):
        model.train()
        for picture in training_data:
            loss_batch(model, loss, picture, optim)
        # do batches and save model after every 5 epochs

        # test loop
        # --------------------------------------------------------------------------

        model.eval()
        total=0
        net_loss=0.0
        with torch.no_grad():
            for picture in validation_data:
                net_loss += loss_batch(model, loss, picture)
                total += 1

        validation_loss = np.sum(np.multiply(net_loss, total)) / total
        # with torch.no_grad():
        #     losses, nums = zip(
        #         *[loss_batch(model, loss, x, y) for x, y in validation_data]
        #     )
        # validation_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, validation_loss)
        #TODO log to Comet as well.

        if (epoch % 10 == 0 || epoch = EPOCHS - 1 ):
            # Save model locally and on comet.
