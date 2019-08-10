import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib import pyplot

def _dataloader(DATA_PATH):     # TODO
    #This is where I load my dataset, and return something that my model can use
    # ------------------------------------------------------------------------------

    data_path = 'Resources/stimuli/'
    train_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=torchvision.transforms.ToTensor())

    training, validation = _split(train_dataset)

    return training, validation


def _split(all_data):    # TODO
    # Split the dataset into a training and validation dataset
    validation = all_data # Some random selection of this.
    training = all_data - validation


    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=64,
    #     num_workers=0,
    #     shuffle=True
    # )

    return training, validation


# ------------------------------------------------------------------------------
# # loading / dataset preprocessing
# tf = transforms.Compose([transforms.ToTensor(),
#                          lambda x: x + torch.zeros_like(x).uniform_(0., 1./args.n_bins)])

# train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True,
#     download=True, transform=tf), batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)

# test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
#     transform=tf), batch_size=args.batch_size, shuffle=False, num_workers=10, drop_last=True)

# # construct model and ship to GPU
# model = Glow_((args.batch_size, 3, 32, 32), args).cuda()
# print(model)
# print("number of model parameters:", sum([np.prod(p.size()) for p in model.parameters()]))

# ------------------------------------------------------------------------------


# # load trained model if necessary (must be done after DataParallel)
# if args.load_dir is not None:
#     model, optim, start_epoch = load_session(model, optim, args)

# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Logging  TODO
# ------------------------------------------------------------------------------

def save_session(model, optim, args, epoch):
    path = os.path.join(args.save_dir, str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    # save the model and optimizer state
    torch.save(model.state_dict(), os.path.join(path, 'model.pth'))
    torch.save(optim.state_dict(), os.path.join(path, 'optim.pth'))
    print('Successfully saved model')

    # if args.comet:
    #     args.experiment.log_metric("Blah", metric, step=time_step)



def load_session(model, optim, args):
    try:
        start_epoch = int(args.load_dir.split('/')[-1])
        model.load_state_dict(torch.load(os.path.join(args.load_dir, 'model.pth')))
        optim.load_state_dict(torch.load(os.path.join(args.load_dir, 'optim.pth')))
        print('Successfully loaded model')
    except Exception as e:
        pdb.set_trace()
        print('Could not restore session properly')

    return model, optim, start_epoch
