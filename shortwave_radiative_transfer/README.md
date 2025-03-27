Open box neural network for shortwave radiative transfer

Author - Henry Schneiderman, henry@pittdata.com

Implemented using PyTorch.

Conda environment setup (installs all Python packages needed by the software in this directory): conda create -n myenv –file package_list.txt

train_network.py - Train neural network

evaluate_network.py - Test and analyze accuracy of a trained network

data_generation.py - Called by train and evaluate to fetch and preprocesses data

network_losses.py - Loss functions for training and testing (e.g., atmospheric layer, clear sky, geographic location)

models/ - Trained models

rnn_ukkonen/ - Peter Ukknonen's RNN for shortwave radiative transfer

