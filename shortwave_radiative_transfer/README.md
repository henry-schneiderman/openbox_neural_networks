Open box neural network for shortwave radiative transfer

Author - Henry Schneiderman, henry@pittdata.com

Implemented using PyTorch.

Conda environment setup (installs all Python packages needed by the software in this directory): conda create -n myenv –file package_list.txt

train_network.py - Train neural network

evaluate_network.py - Test and analyze accuracy of a trained network

data_generation.py - Called by train and evaluate to fetch and preprocesses data

network_losses.py - Loss functions for training and evaluation (e.g., loss as a function of atmospheric layer, geographic location, cosine of solar zenith angle, clear sky vs. full sky)

models/ - Trained models

rnn_ukkonen/ - Peter Ukknonen's RNN for shortwave radiative transfer

Training and testing datasets available at https://zenodo.org/records/15089913


