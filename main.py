from comet_ml import Experiment

import numpy as np
import torch

import argparse
import os
import json
import ipdb

from utils import _dataloader, get_image, load_session
from model import *
from train import fit

use_cuda = torch.cuda.is_available()

# ----------------------------------------------------------------------------------
#                        THIS IS THE AUTOENCODER BRANCH!
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # Where the command line magic happens.
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', default=False, help='to kick off the interactive debugger.')

    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla for logging')
    parser.add_argument('--namestr',type=str,default='neuro_ml',help='additional info in output filename to describe experiments')

    parser.add_argument('--data_dir',type=str,default='Resources/stimuli/',help='path to data')
    parser.add_argument('--load_dir',type=str,default=None,help='use existing model. Load model from _ directory')
    parser.add_argument('--save_dir',type=str,default='weights/',help='directory for saving session')

    parser.add_argument('--batch_size',type=int,default=24 ,help='size of batches')
    parser.add_argument('--n_epochs',type=int, default=2000, help='number of epochs to run for' )

    parser.add_argument('--test_every',type=int, default=200, help='test every _ epochs' )
    parser.add_argument('--save_every',type=int, default=1000, help='save every _ epochs' )
    parser.add_argument('--log_every',type=int, default=5, help='log every _ epochs' )



    args = parser.parse_args()

    if args.debug:
        ipdb.set_trace()

    # Configure Logging.
    # a settings.json file (in gitignore) should be included for logging to comet.
    # ------------------------------------------------------------------------------
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.comet_apikey = data["apikey"]
        args.comet_username = data["username"]
        args.comet_project = data["project"]

    if args.comet:
        experiment = Experiment(api_key=args.comet_apikey, project_name=args.comet_project, workspace=args.comet_username)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    # Because we all like reproducibility (...and also know where we keep our towels)
    # ------------------------------------------------------------------------------
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Obtain and train our model here:
    # ------------------------------------------------------------------------------
    model, optim = get_model()
    if use_cuda:
        model.cuda()

    training_loader, validation_loader = _dataloader(args)

    # load trained model if necessary
    if args.load_dir is not None:
        model, optim, start_epoch = load_session(model, optim, args)
    else:
        start_epoch = 0

    fit(model, training_loader, validation_loader, optim, start_epoch, args)

    args.experiment.end()

    # ------------------------------------------------------------------------------
    # So Long, and Thanks for All the Fish!   >< ((('>    >< ((('>    >< ((('>
    # ------------------------------------------------------------------------------
