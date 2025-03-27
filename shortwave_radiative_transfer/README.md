<h1>Open box neural network for shortwave radiative transfer</h1>

Author - Henry Schneiderman, henry@pittdata.com<br>
Please contact me if you have any questions or feedback

Implemented using PyTorch.

Conda environment setup (installs all Python packages needed by the software in this directory):<br> 
>>conda create -n myenv –file package_list.txt

train_network.py - Train neural network<br>
evaluate_network.py - Test and analyze accuracy of a trained network<br>
data_generation.py - Called by train and evaluate to fetch and preprocesses data<br>
network_losses.py - Loss functions for training and evaluation (e.g., loss as a function of atmospheric layer, geographic location, cosine of solar zenith angle, clear sky vs. full sky)<br>
models/ - Trained models <br>
rnn_ukkonen/ - Peter Ukknonen's RNN for shortwave radiative transfer<br>

Training and testing datasets available at https://zenodo.org/records/15089913



