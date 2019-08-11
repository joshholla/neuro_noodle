import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


def _dataloader(data_path):
    #This is where I load my dataset, and return something that my model can use
    # ------------------------------------------------------------------------------
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=torchvision.transforms.ToTensor())
    augmented_dataset = _augment(train_dataset)

    training, validation = _split(augmented_dataset)
    return training, validation

def _augment(data):
    # Not Implemented TODO
    return data

def _split(train_dataset,
           num_workers=0,
           valid_size=0.1,
           sampler=SubsetRandomSampler,
           args):
    batch_size = args.batch_size
    num_train=len(train_dataset)
    indices= list (range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = sampler(train_idx)
    valid_sampler = sampler(valid_idx)

    training = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    validation = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    return training,validation


def get_image(x):
    output = ToPILImage()
    return output(x)

# Future TODO:
# One more thing I could do is normalize the data, and check for the mean and
# SD. Get to it later if I can. Would be a good analysis of the dataset.

# https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
# https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/4?u=ajong
# In case I want to save the image names in the face, and train a classifier for
# funsies.

# ------------------------------------------------------------------------------
# # construct model and ship to GPU
# model = Glow_((args.batch_size, 3, 32, 32), args).cuda()
# print(model)
# print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Logging  TODO
# ------------------------------------------------------------------------------

def save_session(model, optim, args, epoch):
    path = os.path.join(args.save_dir, str(epoch)) # appending epoch number to
    # the file, so I know what epoch the saved weights are from.
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

    #save to Comet Asset Tab
    if args.comet:
        args.experiment.log_asset(file_path= os.path.join(path, 'model.pth'), file_name='autoencoder_model.pth' )
        args.experiment.log_asset(file_path= os.path.join(path, 'optim.pth'), file_name='autoencoder_optim.pth' )


def load_session(model, optim, args):
    try:
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        ipdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch
