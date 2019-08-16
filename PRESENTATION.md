Walkthrough - master branch (AutoEncoder)
==

This is the `master` branch, and it contains code that trains an autoencoder on our dataset. 
The idea is to train an autoencoder, and in doing so obtain a good latent representation for images in the dataset. 

The `Plotting` folder contains an ipynb that lets you plot a clustering from the data.

---

The codebase was written with readability in mind. This file will serve as an overview. 

Let's dive in, shall we? 

The project file structure is as follows:
```
.
├── LICENSE.md
├── Plotting
│   ├── clustering.ipynb
│   ├── umap_pixel.png
│   └── umap_pixel_2.png
├── README.md
├── main.py
├── model.py
├── Resources
│   └── stimuli
│       └── <data>
├── requirements.txt
├── run.sh
├── scratch
│   ├── Dataloading.ipynb
│   └── thoughts.md
├── settings.json
├── train.py
└── utils.py
```



---


### [main.py](https://github.com/joshholla/neuro_noodle/blob/master/main.py)
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

The next section deals with checking for and parseing the `settings.json` file, to deal with logging to comet.

Then I manually set the random seeds ([to 42](https://en.wikipedia.org/wiki/42_(number)#The_Hitchhiker's_Guide_to_the_Galaxy)) for reproducibility.  

After that we instantiate our model, and if we are connected to a GPU, we send it to the GPU.  

Next we call `_dataloader`, and obtain data loaders both for training data and validation data.

Finally we call `fit()` from `train.py`, to train our model.

### [utils.py](https://github.com/joshholla/neuro_noodle/blob/master/utils.py)
After importing the required files, we have the `_dataloader` function, that loads the dataset using Pytorch's Dataloader, and splits our data into a 90-10 split of training and validation data. `_augment` wasn't implemented. I planned on using some transforms on the images to give me more data to play with, but in the end elected not to for this iteration.

There is a function called `get_image()` which is meant to be able to obtain the original image. However, I commented it out since we do not want the data set images appearing in public.

Finally, we also have functions that write or pull the session state to disk. If comet logging is enabled via the command line, the model state and optimizer state will be pushed to the experiment's asset tab as well. 

### [train.py](https://github.com/joshholla/neuro_noodle/blob/master/train.py)
The main contribution of this file, is to train our neural network. 
In the `fit()` function, it sets up the training, taking care to only update weights and maintain gradients only when working on training data. We use [Binary Cross Entropy Loss](https://pytorch.org/docs/stable/nn.html#bceloss) for checking the difference between the output of the autoencoder, and the original image. 

We also have code in there to assert that loss doesn't stay at zero, a tip I picked up from [reading about debugging ML code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765).

### [model.py](https://github.com/joshholla/neuro_noodle/blob/master/model.py)
Our model is a simple autoencoder. We use fully connected networks, going down to a latent layer with 40 neurons at the output of the encoder, before building back to the originaln size with the decoder.

We also use the Adam optimizer, and a learning rate of 0.001.

### Plotting/[clustering.ipynb](https://github.com/joshholla/neuro_noodle/blob/master/Plotting/clustering.ipynb):
This contains code for loading the dataset and manipulating it with sklearn UMAP. It was developed by looking at the [documentation](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) for UMAP.


### Miscellaneous

+ [LICENSE.md](https://github.com/joshholla/neuro_noodle/blob/master/LICENSE.md)
this repo is released with an MIT license

+ In order to log results to my `comet.ml` repository, I have a configuration stored in a `settings.json` file.
The file looks something like this:
```
{"username":"<username>", "apikey":"<key>", "restapikey":"<key>", "project":"neuromlnoodle"}
```

+ [run.sh](https://github.com/joshholla/neuro_noodle/blob/master/run.sh)
This file is a shell script that makes running experiments easy to parallelize, and prevents having to type in long sequences of text when adding arguments to my experimental code.  
It also gets logged, so that helps with reproducibility.

+ The `scratch/` folder was used to run a python notebook with my intermediate steps while developing the project.  

+ `requirements.txt`
This contains the project dependancies.

+ Data is stored in `Resources/stimuli`

---


[Take me back to the Welcome page](https://joshholla.github.io/neuro_noodle/)