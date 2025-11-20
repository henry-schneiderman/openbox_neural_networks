<h1>Open box neural network for shortwave radiative transfer</h1>

Author - Henry Schneiderman, henry@pittdata.com<br>
Please contact me for any questions or feedback

H. Schneiderman. "An Open Box Physics-Based Neural Network for Modeling Shortwave Radiative Transfer." To appear in *Artificial Intelligence for the Earth Systems* (AIES)

---
Conda environment setup (installs all required Python packages):</br> conda create -n myenv –file package_list.txt

Training and testing datasets available at https://zenodo.org/records/15089913

---
evaluate_network.py - For testing and analyzing trained networks listed below<br>

train_network.py - "Baseline" neural network described in Sections 4.b - 4.f<br>

train_network_3.py - Modified network allowing all atmospheric constituents to influence all spectral channels as described in Section 4.i. <br>

train_network_4.py - Modified network containing a scattering component that processes an alternative set of inputs (the masses of the atmospheric constituents, temperature, and pressure rather than the optical depths of the atmospheric constituents) as described in Section 4.j.<br>

train_network_5.py, train_network_6.py - Modified neural networks that have only 28 and 14 spectral channels, instead of 42, as described in Section 4.g. <br>

train_network_7.py - Modified neural network that replaces the transmissivity and scattering components with an approach that attempts to learn each layer's radiative properties without embedding Beer's Law as described in Section 4.h. <br>

data_generation.py - Fetches and preprocesses data. Used by training and evaluation<br>

network_losses.py - Loss functions for training and evaluation (e.g., loss as a function of atmospheric layer, geographic location, cosine of solar zenith angle, clear sky vs. full sky)<br>

models/ - Trained model weights <br>

rnn_ukkonen/ - Peter Ukkonen's RNN for shortwave radiative transfer<br>

rnn_ukkonen/data_processing/reformat_data.py - Reformats all data used by openbox network to the format used by RNN<br>

rnn_ukkonen/models/ - Trained RNN weights







