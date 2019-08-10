Neuro Noodle
==

This repository is my exploratory dive on the data set that was provided to us at [this site](https://https://www.dropbox.com/sh/v0t0s2qrafmfziu/AACg_xr2r7VF4YNvejuPu-9pa?dl=0)


This file documents my journey with the data. My findings are found in findings.ipynb

On observing the the data, it appears to be a set of faces, each showing a range of emotions, ranging from happy to sad. 

<Insert some Pictures here>

On seeing that it's faces that show a few emotions, and going by the fact that the project is based on identifying emotions, I thought it would be a fun exercise to obtain a latent representation of the emotions in the faces, and try to map out an interactive plot.     


It would also be fun to take an un-seen face, and generate the emotion based on that ( using a GAN or a VAE)
I haven't worked with either yet, so it would be a fun exercise to go through learning it that way.   

Section on why I think that GANs are probably super cool, but not what I want to use for the job.


So first of all. Let's set expectations on what I want to achieve in the process:

1.  First off, I want to read in all the faces, and form a latent representation. The example paper uses PCA. but I think a cool thing to do, would be to use Variational AutoEncoders to check this out.

    + Data Loading for the faces that I've been given.
    + This would be where I do my data pre-processing. Split up the data set into a training validation and testing set.
    + Probably do some transfer learning from an existing model? To get decent accuracy i.e. ResNet?
    + Then get the encoder. Learn the Variational AutoEncoder form.
    + Upload Training data to Cometml.

2. Once I have latent space representations for this, I want to go about plotting my datapoints, probably use tSNE, and see whether I get a good grouping of my data that way. 

3. Once I am happy with the representations, I'll take a test set and work with just the base face, and try to set it's emotion. Compare this with the original data-set, and that can serve as an error function.  







