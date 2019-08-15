from comet_ml import Experiment

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import json

import tqdm
from tqdm import tqdm
import ipdb

from utils import _dataloader
from model import *
from train import *

use_cuda = torch.cuda.is_available()

# ----------------------------------------------------------------------------------
#                           THIS IS THE CLASSIFIER BRANCH!
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    # Where the command line magic happens
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', default=False, help='to kick off the interactive debugger')

    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla for logging')
    parser.add_argument('--namestr',type=str,default='neuro_ml',help='additional info in output filename to describe experiments')

    parser.add_argument('--data_dir',type=str,default='Resources/classify/',help='path to data')
    parser.add_argument('--load_dir',type=str,default=None,help='use existing model, send local path to saved model')
    parser.add_argument('--save_dir',type=str,default='weights/',help='directory for saving session')

    parser.add_argument('--batch_size',type=int,default=24 ,help='size of batches')
    parser.add_argument('--n_epochs',type=int, default=40, help='number of epochs to run for' )

    parser.add_argument('--test_every',type=int, default=200, help='test every _ epochs' )
    parser.add_argument('--save_every',type=int, default=500, help='save every _ epochs' )
    parser.add_argument('--log_every',type=int, default=5, help='log every _ epochs' )

    parser.add_argument("--rejig_data", action='store_true', default=False, help='to move the data around for classification')

    args = parser.parse_args()

    if args.debug:
        args.use_logger = False
        ipdb.set_trace()


    # Call only once. Moves data into new folders and classes. Suitable for
    # happy vs sad classification
    # ------------------------------------------------------------------------------
    if args.rejig_data:
        make_classification_dataset()
        print("Dataset has been re-located for classification. Please remove --rejig_data flag in future runs")
        break

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


    # Let's use a pre-trained classification model like ResNet!
    # ------------------------------------------------------------------------------

    model_finetune, input_size = get_pretrained_model()
    if use_cuda:
        model_finetune.cuda()

    dataloaders = _dataloader(args, input_size)

    parameters_to_update=model_finetune.parameters()
    optimizer_finetune= optim.SGD(parameters_to_update, lr= 0.001, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    model_finetune, hist = fit(model_finetune, dataloaders, criterion, optimizer_finetune, args)

    args.experiment.end()


    # ------------------------------------------------------------------------------
    # So Long, and Thanks for All the Fish!    >< ((('>    >< ((('>    >< ((('>
    # ------------------------------------------------------------------------------
