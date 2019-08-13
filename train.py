import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms, utils, models
import matplotlib.pyplot as plt

import numpy as np
import ipdb
import tqdm
from tqdm import tqdm
import os
import copy

from model import *
from utils import *

use_cuda = torch.cuda.is_available()

def fit(model, dataloaders, criterion, optimizer, args):
    num_epochs = args.n_epochs
    validation_accuracy_history = []
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc= 0.0

    for epoch in tqdm(range(num_epochs)):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} loss = {epoch_loss}")
            print(f"{phase} accuracy = {epoch_acc}")
            if args.comet:
                args.experiment.log_metric(f"{phase} loss", epoch_loss, step=epoch)
                args.experiment.log_metric(f"{phase} accuracy", epoch_acc, step=epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_model_weights = copy.deepcopy(model.state_dict())
                best_acc = epoch_acc
            if phase == 'val':
                validation_accuracy_history.append(epoch_acc)

    print(f"Best validation accuracy= {best_acc}")
    if args.comet:
        args.experiment.log_metric(f"Best validation accuracy", best_acc)

    # send back the best model seen so far:
    model.load_state_dict(best_model_weights)
    save_session(model, optim, args)
    return model, validation_accuracy_history
