Welcome!
==

If this is your first time here, please refer to the repository's associated web page [here](https://joshholla.github.io/neuro_noodle/). (It contains my findings and presents my code)

This is the `master` branch, and it contains code that trains an autoencoder on our dataset. The idea is to find latent representations for the images. 

The `Plotting` folder contains an ipynb that lets you plot a clustering from the data.

## Getting Started

To get started using this repository, please install dependancies using pip:
> $ pip install -r requirements.txt   


The project file structure should be as follows:
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

Add the data at `Resources/stimuli`

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


## Other Branches

Don't forget to look at the other two branches in this repository:
- [The classification branch](https://github.com/joshholla/neuro_noodle/tree/classification): Where I shuffled the dataset and built the emotion classifier
- [The gh-pages branch](https://github.com/joshholla/neuro_noodle/tree/gh-pages): for this repository's website and 'homework' presentation.

Enjoy!

(PS - if you're wondering why I named this noodling, I'm refering to this [definition](https://www.dictionary.com/browse/noodling))