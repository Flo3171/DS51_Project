# ds51-project

A project for the DS51 class, where we had to use SPARQL to query information about species,
and train a small neural network model to recognize those species in images.

The twist for this small neural network is that it is trained to recognize a vector of caracteristics for each of the species.
We then find the species in our dataset that has the closest vector (using euclidean distance).

You will find in this repository:
- `preliminary_analysis.ipynb`, a notebook with a few analysis steps made on the dataset
- `animal_ontologie.rdf`, an owl graph containing information about the different species we are interested in
- `load_ontology.ipynb`, a notebook where we load the above graph into python and extract the information we need
- `tensorflow.ipynb`, a notebook where we use tensorflow to train and run a neural network
- `main.py`, a compilation of the above two notebooks into a working python project

You will *not* find in this repository the dataset, which is a subset of [ImageNet](https://www.imagenet.org/), with only the following classes:
```
n01443537
n01484850
n01537544
n01614925
n02114367
n02133161
```

This dataset should be placed within the `dataset/` folder.

To run the code, simply do `python code/main.py`.
It will load the neural network and the owl graph, and predict the species of the validation split of the dataset.

If you want to give a shot to training the neural network, run `python code/main.py train`.
