Welcome to the Classification branch!
==

If this is your first time here, please refer to the repository's associated web page [here](https://joshholla.github.io/neuro_noodle/). (It contains my findings and presents my code)

This repository deals with the building of a classifier. It classifies faces in the dataset as happy or sad.  
I reason that a good classifier might be useful in training a computer to view fMRI scans and infer emotions from them.

## Getting Started

To get started using this repository, please install dependancies using pip:
> $ pip install -r requirements.txt   


The project file structure should be as follows:
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

Add the data to `Resources/stimuli`

You first need to run a command to re-jig the dataset. This is done by running 
```
python main.py --rejig_data
```
Once this is done, the classifier can be trained to detect emotion from the dataset.

## Logging

If you plan to log results to your `comet.ml` repository, please add and populate a `settings.json` file.

Your file should look something like this:
```
{"username":"<username>", "apikey":"<key>", "restapikey":"<key>", "project":"neuromlnoodle"}
```

## Running the program

In order to launch the program, you can launch `run.sh` or run using python3

```
python main.py --args
```
The various arguments and their functionality are listed in `main.py` file.

The `.ipynb` notebooks can be moved to their parent directory and run interactively.  

Logs of the experiments that I ran, are available [here](https://www.comet.ml/joshholla/neuromlnoodle/view/). 


Enjoy!