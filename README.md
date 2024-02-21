# Deep Learning based Multi-Objective Deformable Image Registration
This repository contains code for applying Multi-objective learning described in paper [**Multi-objective Learning Using HV Maximization**](https://www.springerprofessional.de/en/multi-objective-learning-using-hv-maximization/24601248) to deformable image registration. The pdf is available [here](https://pure.tudelft.nl/ws/portalfiles/portal/150437476/978_3_031_27250_9_8.pdf). 

## Set up
The repository uses `poetry` as a package manager. To set up the environment run the following commands:

- ``curl -sSL https://install.python-poetry.org | python3 -`` to install `poetry`
- ``poetry config virtualenvs.in-project true`` to have poetry install the .venv in this folder
- ``poetry install`` in this working directory to setup your virtual environment
- activate the environment

Additionally, you may need to run `pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117` to install torch separately in the .venv.

## Structure of the repository

├── data_preparation    # codes to affine register the MRI scans, make pairs, and generate csv file

├── default_configs    # configuration files containing default hyperparameters used for different experiments

├── functions    # multiple functions used for HV maximization based MO learning

├── mo_optimizers    # different MO optimizers tested in the MO learning code. Note: MO optimizers other than HV maximization and linear scalarization may not be compatible with the curretn repo.

├── problems

│   ├── DTLZ    # 3D MO benchmark problem, DTLZ

│   ├── genmed    # 3D MO benchmark problem, genmed

│   ├── mo_regression    # MO regression problem from MO learning paper

│   ├── modir    # 2D MO DIR problem with MNIST dataset. *Note: compatibility hasn't been tested with recent changes*

│   ├── modir3d    # 3D MO DIR with brachytherapy data

│   ├── ZDT    # 2D MO benchmark problem, ZDT

├── runs    # experiment outputs will save here

├── utilities    # extra stuff

├── cli.py    # interpret command-line arguments

├── config.py    # contains description of all configurable variables

├── main.py    # main file

├── net_ensemble.py    # contains DeepEnsemble class for creating a neural network set class, and KheadEnsemble class for creating a single neural network with multiple heads

├── runtime_cache.py    # keeps track of run time variables e.g., number of epochs

└── setup.py    # main.py calls `setup_train` and `setup_test` methods from this to run an experiment

├── testing.py    # wrapper to run inference for a particular experiment

├── training.py    # MO training code

**Adding a new problem to the MO framework** is straight-forward. Create a new folder in the `problems` folder with the following structure:

├── new_problem

│   ├── data.py    # must contain a `get_dataset` function that returns a pytorch `Dataset` object

│   ├── inference.py    # to be used for inference on testing data

│   ├── losses.py    # must contain a `Loss` class that takes `loss_name_list` as an input to initialize MO loss

│   ├── model.py    # must contain a `get_network` function that returns a pytorch `nn.Module` object


In the `config.py`, add the problem name in the list of possible problem name as follows:
`PROBLEM_NAME: Literal["mo_regression", "modir", "modir3d", "dtlz", "genmed", "zdt", "new_problem"]`

Similarly, a new model, a new dataset, or a new loss can be added to an existing problem and modifying `get_network`, `get_dataset`, and `Loss`, respectively in the problem folder.

## Train a MO DIR model using VoxelMorph as DIR network
VoxelMorph is a DIR neural network proposed in [*VoxelMorph: A Learning Framework for Deformable Medical Image Registration*](https://ieeexplore.ieee.org/abstract/document/8633930). We use the original code for VoxelMorph available from https://github.com/voxelmorph/voxelmorph and modify neural network architecture to create its MO version. The modified neural network is in `problems/modir3d/mo_voxelmorph.py`.
To train MO DIR using this model, run the following:

`CUDA_VISIBLE_DEVICES=0 python -m main --env-file ./default_configs/modir3d_hv_mo_voxelmorph`

Also, check out other default configs to run different problems. Alternatively, create a new config file where specific variables are modified.

*Note: Training will create an output folder structure in `runs` folder based on the variable `EXPERIMENT_NAME`.*

## Testing MO DIR
To test all the models in an experiment for a test dataset, create a config file similar to `default_config/testing/test_try` and run the following:

`CUDA_VISIBLE_DEVICES=0 python -m main --test-env-file ./default_configs/testing/test_try`


