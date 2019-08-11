import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from utils import *
import json
import tqdm
import ipdb
from tqdm import tqdm
from comet_ml import Experiment
from model import *
from train import *

use_cuda = torch.cuda.is_available()

# ------------------------------------------------------------------------------
# ADD ASSERTS AND TESTING TOO!
# ------------------------------------------------------------------------------

if __name__ = "__main__":
    # Where the command line magic happens
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', default=False, help='to prevent logging (even to disk), when debugging.')
    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla for logging')
    parser.add_argument("--use_logger",action='store_true',default=False,help='to log or not to log (that is the question)')
    parser.add_argument('--namestr',type=str,default='neuro_ml',help='additional info in output filename to describe experiments')
    parser.add_argument('--load_dir',type=str,default=None,help='use existing model, send local path to saved model')
    parser.add_argument('--save_dir',type=str,default='weights/',help='directory for saving session')

    args = parser.parse_args()

    if args.debug:
        args.use_logger = False
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
        experiment = Experiment(api_key=args.comet_apikey,project_name=args.comet_project,auto_output_logging="None",workspace=args.comet_username,auto_metric_logging=False,auto_param_logging=False)
        experiment.set_name(args.namestr)
        args.experiment = experiment

    # Because we all like reproducibility (...and also know where we keep our towels)
    # ------------------------------------------------------------------------------
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Training here:
    # ------------------------------------------------------------------------------
    model, optim = get_model()
    if use_cuda:
        model.cuda()
    data = _dataloader()

    # load trained model if necessary
    if args.load_dir is not None:
        model, optim, start_epoch = load_session(model, optim, args)

    epoch = train(model, data) # Gotta return epoch and stuff at the end?
    model.save_session(model, optim, epoch) # Log to disk and to comet.


    # TODO: I might want to sample the data at some point, and physically look
    # at what I'm seeing. Write something for that. Call intermittently.


    # ------------------------------------------------------------------------------
    # So Long, and Thanks for All the Fish!   >< ((('>
    # ------------------------------------------------------------------------------
