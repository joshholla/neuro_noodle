#!/usr/bin/env bash

#set -x
#echo "Getting into the script"

# Script to run the experiment in this repository.
# Can also be used to change up hyperparameter settings, and run in parallel if
# desired.

python main.py --comet --namestr="Train_Classifier" --n_epochs=25