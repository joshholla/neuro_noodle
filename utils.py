import os
import glob
import shutil
import numpy as np
import torch
import torchvision
import PIL
import random
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils, datasets
from matplotlib import pyplot
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


# ----------------------------------------------------------------------------------
#                         CREATE A CLASSIFICATION DATASET
# ----------------------------------------------------------------------------------

def make_classification_dataset():
    os.mkdir('Resources/classify')
    os.mkdir('Resources/classify/train')
    os.mkdir('Resources/classify/val')
    os.mkdir('Resources/classify/train/happy')
    os.mkdir('Resources/classify/val/sad')
    os.mkdir('Resources/classify/train/happy')
    os.mkdir('Resources/classify/val/sad')
    os.mkdir('Resources/classify/train/sad')
    os.mkdir('Resources/classify/val/happy')
    for filepath in glob.iglob('Resources/stimuli/*/*.JPG'):
        source_directory=filepath
        if '_Sa_' in source_directory:
            if random.random() < 0.1:
                shutil.move(source_directory, 'Resources/classify/val/sad')
            else:
                shutil.move(source_directory, 'Resources/classify/train/sad')
        elif '_Ha_' in source_directory:
            if random.random() < 0.1:
                shutil.move(source_directory, 'Resources/classify/val/happy')
            else:
                shutil.move(source_directory, 'Resources/classify/train/happy')


# ----------------------------------------------------------------------------------
#                               DATALOADING
# ----------------------------------------------------------------------------------

def _dataloader(args, input_size):
    #This is where I load my dataset, and return something that my model can use
    # ------------------------------------------------------------------------------

    # Do data augmentation and normalization for training
    # Only do normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

    return dataloaders_dict

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------

def save_session(model, optim, args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # save the model state
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
    print('Successfully saved model')

    #save to Comet Asset Tab
    if args.comet:
        args.experiment.log_asset(file_data= args.save_dir+'/model.pth', file_name='classification_model.pth' )
