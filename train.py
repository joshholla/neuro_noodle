import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils

import numpy as np
import ipdb
import tqdm
from tqdm import tqdm


from model import *
from utils import save_session, get_image


def loss_calc(model, loss_function, image, optim=None):
    loss = loss_function(model(image), image)
    # so that we don't train on validation data
    if optim is not None:
        loss.backward()
        optim.step
        optim.zero_grad()

    # Testing that loss isn't stuck at zero
    # --------------------------------------------------------------------------

    assert loss.item() !=0, "Loss is staying at Zero! (T_T)  "
    return loss.item()

def show_image(model, image, args):
    reconstruct = model(image)
    if args.comet:
        args.experiment.log_image(get_image(reconstruct), name= epoch)


def fit (model, training_data, validation_data, optim, start_epoch, args):
    loss = torch.nn.BCELoss()
    for epoch in tqdm(range(start_epoch, start_epoch+args.n_epochs)):
        # training loop
        # ----------------------------------------------------------------------
        model.train()
        for data, _ in training_data:
            for _, picture in enumerate(data):
                picture = picture.view(picture.size(0), -1)
                if use_cuda:
                    picture = picture.cuda()
                training_loss = loss_calc(model, loss, picture, optim)
                if args.comet:
                    args.experiment.log_metric("Training Loss", training_loss)

        # test loop
        # ----------------------------------------------------------------------
        if ( (epoch+1) % args.test_every == 0 ):
            model.eval()
            with torch.no_grad():
                for data, _  in validation_data:
                    for _, picture in enumerate(data):
                        picture = picture.view(picture.size(0), -1)
                        if use_cuda:
                            picture = picture.cuda()
                        val_loss = loss_calc(model, loss, picture)
                        if args.comet:
                            args.experiment.log_metric("Validation Loss",
                                                       val_loss, step=epoch)
                        image = picture

        if ((epoch+1) % args.save_every == 0):
            save_session(model, optim, args,  epoch)

        last_epoch = epoch
    save_session(model, optim, args, last_epoch)
