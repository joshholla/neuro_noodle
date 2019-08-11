import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import ipdb
import argparse
import time

from model import *
from utils import *

def get_model():
    model = Autoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer

def loss_batch(model, loss_function, image, optim=None):
    loss = loss_function(model(image), image)
    # so that we don't train on validation data
    if optim is not None:
        loss.backward()
        optim.step
        optim.zero_grad()

    return loss.item()

def show_image(model, image):
    reconstruct = model(image)
    if args.comet:
        args.experiment.log_image(get_image(reconstruct), name= epoch)


def fit (model, training_data, validation_data, optim, start_epoch, args):
    loss = torch.nn.BCELoss()
    for epoch in tqdm(range(start_epoch, start_epoch+args.n_epochs)):
        # training loop
        # ------------------------------------------------------------------------------
        model.train()
        for picture in training_data:
            loss_batch(model, loss, picture, optim)

        # test loop
        # --------------------------------------------------------------------------
        if ( (epoch+1) % args.test_every == 0 ):
            model.eval()
            total=0
            net_loss=0.0
            with torch.no_grad():
                for picture in validation_data:
                    net_loss += loss_batch(model, loss, picture)
                    total += 1
                    image = picture

            validation_loss = np.sum(np.multiply(net_loss, total)) / total

            if ((epoch+1) % args.log_every == 0 ):
                print(epoch, validation_loss)
                if args.comet:
                    args.experiment.log_metric("Validation Loss", validation_loss, step= epoch)

                # also sample data and see what the reconstruction looks like
                show_image(model, image)

        if ((epoch+1) % args.save_every == 0):
            model.save_session(model, optim, epoch)

        last_epoch = epoch
    model.save_session(model, optim, last_epoch)
