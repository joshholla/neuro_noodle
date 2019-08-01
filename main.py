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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ = "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true', default=False, help='to prevent logging (even to disk), when debugging.')
    parser.add_argument("--comet", action='store_true', default=False, help='to use https://www.comet.ml/joshholla for logging')
    parser.add_argument("--use_logger",action='store_true',default=False,help='to log or not to log (that is the question)')
    parser.add_argument('--namestr',type=str,default='neuro_ml',help='additional info in output filename to describe experiments')


    args = parser.parse_args()

    if args.debug:
        args.use_logger = False
        ipdb.set_trace()


    # Check for a settings.json file for comet logging
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




    # Write code to train here.



    # Write this in utils, for logging.

    # if args.comet:
    #     args.experiment.log_metric("Blah", metric, step=time_step)
