Walkthrough - classification branch
==

This is the `classification` branch, and it contains code that trains a classifier on our dataset. 

The idea is that techniques like this will come in handy when we're dealing with fMRI images and want to identify emotions via those outputs. 

---

The code (and commit messages) have been written with readability in mind. 
This document serves as a starting point and overall guide. 
All the hyperlinks go to the files that they reference. 

The project file structure is as follows:
```
.
├── LICENSE.md
├── README.md
├── main.py
├── model.py
├── Resources
│   └── stimuli
│       └── <data>
├── requirements.txt
├── run.sh
├── scratch
│   └── Classynoodle.ipynb
├── settings.json
├── train.py
└── utils.py
```


---


### [main.py](https://github.com/joshholla/neuro_noodle/blob/classification/main.py)
This file is meant to be run in the command line. 

Like most files, it starts off with a bunch of imports.

Next it deals with parseing the arguments entered from the command line (using [argparse](https://docs.python.org/3/library/argparse.html)). This can be used to add the following arguments:

| Argument | Function         | 
| -------- | ---------------- | 
| debug    | helps to kick off the interactive debugger [ipdb](https://pypi.org/project/ipdb/) 
| comet   | tells the program to use remote logging. the settings file is configured to send logs to [my page](https://www.comet.ml/joshholla/neuromlnoodle/view/) |
| namestr   | this is the name that we can use to find the experiment again when running it on comet.   |
| data_dir | is the path to where data is stored |
| load_dir | if we are using an existing model (picking back up with training), can load the model from _ directory |
| save_dir | is the directory to where sessions ( model weight and optimizer weights ) can be written to|
| batch_size | choose the size of mini batches to return in the dataloader |
| n_epochs | this is the number of epochs that we want to run the program for |
| test_every | run the testing subroutine every _ epochs |
| save_every | save session every _ epochs |
| log_every | log every _ epochs |
| rejig_data | creates a new dataset by re-arranging the existing one. New dataset is amenable to running a classifier on it |

The next section deals with checking for and parsing the `settings.json` file, to deal with logging to comet.

Then I manually set the random seeds ([to 42](https://en.wikipedia.org/wiki/42_(number)#The_Hitchhiker's_Guide_to_the_Galaxy)) for reproducibility.  

After that we instantiate our model, and if we are connected to a GPU, we send it to the GPU.  

Next we call `_dataloader()`, and obtain both data loaders for training data and validation data.

Finally we call `fit()` from `train.py`, to re-train our model.

### [utils.py](https://github.com/joshholla/neuro_noodle/blob/classification/utils.py)

`make_classification_dataset` creates a datastructure that looks like:

```
.
└── Resources
    └── classify
        ├── train
        │   ├── happy
        │   └── sad
        └── val
            ├── happy
            └── sad
```

This function then moves files with `_Sa_` into the sad folders, with 10% of them going to the validation branch and the others to training. It does the same with files that have `_Ha_` in them to the happy folders.

Then we have the `_dataloader` function that performs some data augmentation (like Random Horizontal flips) and Normalizing on the training set. We take care not to do any augmentation for the validation set. 

We take advantage of Pytorch's DataLoaders to return a dictionary that has the training and validation dataloaders.

Finally, we also have functions that write the model state to disk. If comet logging is enabled via the command line, the model weights will be pushed to the experiment's asset tab as well.


### [train.py](https://github.com/joshholla/neuro_noodle/blob/classification/train.py)
The main contribution of this file, is to train our neural network. 
In the `fit()` function, it sets up the training of the modified ResNet18 network, taking care to only update weights and maintain gradients only when working on training data. We use [Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) for our loss function. 

We also have code in there to assert that loss doesn't stay at zero, a tip I picked up from [reading about debugging ML code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765).

This function saves and returns the model that gave us the best validation accuracy because we want the model with the best accuracy on the validation set to be used for predictions.

This 'best' model is logged locally and optionally to comet.

### [model.py](https://github.com/joshholla/neuro_noodle/blob/classification/model.py)
We want to take advantage of using pre-trained models. This theoretically lets us short-cut a lot of the training that we would normally have to do on a model that began from scratch. 
We replace the last layer of the model with a fully connected layer that has 2 outputs (because we have 2 classes)


We also use [Stochastic Gradient Descent](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) and a learning rate of 0.001 and momentum of 0.9.

### scratch/[Classynoodle.ipynb](https://github.com/joshholla/neuro_noodle/blob/classification/scratch/Classynoodle.ipynb)
This notebook was run while I was developing the project. 
It could be likened to a scratchpad of sorts.


### Miscellaneous

+ [LICENSE.md](https://github.com/joshholla/neuro_noodle/blob/classification/LICENSE.md)
this repo is released with an MIT license

+ In order to log results to my `comet.ml` repository, I have a configuration stored in a `settings.json` file.
The file looks something like this:
```
{"username":"<username>", "apikey":"<key>", "restapikey":"<key>", "project":"neuromlnoodle"}
```

+ [run.sh](https://github.com/joshholla/neuro_noodle/blob/classification/run.sh)
This file is a shell script that makes running experiments easy to parallelize, and prevents having to type in long sequences of text when adding arguments to my experimental code.  

+ `requirements.txt`
This contains the project dependancies.

+ Data is stored in `Resources/stimuli`

---


[Take me back to the Welcome page](https://joshholla.github.io/neuro_noodle/)