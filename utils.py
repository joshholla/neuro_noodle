import os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToPILImage


# ----------------------------------------------------------------------------------
#                               DATALOADING
# ----------------------------------------------------------------------------------

def _dataloader(args):
    # This is where I load the dataset and return something that the model can use
    # ------------------------------------------------------------------------------
    train_dataset = torchvision.datasets.ImageFolder(root=args.data_dir,transform=torchvision.transforms.ToTensor())
    augmented_dataset = _augment(train_dataset)

    training, validation = _split(augmented_dataset, args)
    return training, validation

def _augment(data):
    # Not Implemented.
    return data

def _split(train_dataset,args, num_workers=0, valid_size=0.1, sampler=torch.utils.data.sampler.SubsetRandomSampler):
    # This is where I split data into training and validation sets.
    # ------------------------------------------------------------------------------
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


# Don't use this function as we don't want the dataset images in public.
def get_image(x):
#     x.detatch().numpy()
#     output = ToPILImage()
#     return output(x)
    return x

# ----------------------------------------------------------------------------------
#                               LOGGING
# ----------------------------------------------------------------------------------

def save_session(model, optim, args, epoch):
    # appending epoch number to the file, so I know what epoch the saved weights
    # are from.
    # ------------------------------------------------------------------------------
    path = os.path.join(args.save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    # ------------------------------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

    #save to Comet Asset Tab
    if args.comet:
        args.experiment.log_asset(file_data= args.save_dir+'/'+str(epoch)+'/' +'model.pth', file_name='autoencoder_model.pth' )
        args.experiment.log_asset(file_data= args.save_dir+'/'+str(epoch)+'/' +'optim.pth', file_name='autoencoder_optim.pth' )


def load_session(model, optim, args):
    # Bring the model back, and restart. (With exception handling)
    # ------------------------------------------------------------------------------
    try:
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        ipdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch
