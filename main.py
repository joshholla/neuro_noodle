import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import utils
import json
import tqdm
import ipdb
from tqdm import tqdm
from comet_ml import Experiment

use_cuda = torch.cuda.is_available()


# train is Icky, and needs implementation: TODO

# def train(data_set, epochs=EPOCHS, batch_size=BATCH_SIZE):
#     loss_func = torch.nn.BCELoss() # USE THIS LOSS!!
#     for epoch in tqdm(range(epochs)):
#         for ii, batch_raw in enumerate(tqdm(data_loader)):
#             optim.zero_grad()

# ADD ASSERTS AND TESTING TOO!

# ------------------------------------------------------------------------------

if __name__ = "__main__":
    """Where the command line magic happens"""
    # ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', default=False, help='to prevent logging (even to disk), when debugging.')
    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla for logging')
    parser.add_argument("--use_logger",action='store_true',default=False,help='to log or not to log (that is the question)')
    parser.add_argument('--namestr',type=str,default='neuro_ml',help='additional info in output filename to describe experiments')

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
    model = Autoencoder()
    if use_cuda:
        model.cuda()

    optim = Adam(model.parameters(), lr = 0.001)

    data = _dataloader()
    train(data_set=data)
    model.save_json() # Log to disk and to comet.


    # TODO: I might want to sample the data at some point, and physically look
    # at what I'm seeing. Write something for that. Call intermittently.
    # Write this in utils, for logging.

    # if args.comet:
    #     args.experiment.log_metric("Blah", metric, step=time_step)




    # So Long, and Thanks for All the Fish!
