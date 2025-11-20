"""
For evaluating and analyzing performance of open box networks trained by
train_network.py, train_network_3.py, ..., training_network_7.py

Note: contains slightly altered versions of some of the classes in
training_network.py

Author: Henry Schneiderman, henry@pittdata.com
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from netCDF4 import Dataset
import time

# Specify network to evaluate by importing it as "tn"
#import train_network_3 as tn
import train_network as tn
import data_generation
import network_losses as nl

# Elapsed time for various computations
t_total = 0.0
t_optical_depth = 0.0
t_scattering = 0.0
t_multireflection = 0.0
t_propagation = 0.0

t_adding_doubling = 0.0
t_division = 0.0
t_first_operation = 0.0
t_surface = 0.0

class ScatteringTiming(nn.Module):
    """ 
    The same as Scattering in train_network.py except 
    final computation normalizes outputs such that:
     
        e_t_direct + e_r_direct + e_a_direct + t_direct = 1.0
        e_t_diffuse + e_r_diffuse + e_a_diffuse + t_diffuse = 1.0
        
    MultiReflectionTiming (below) assumes this normalization
        
    Whereas, Scattering and Multireflection in training_network.py 
    use the following normalization:
        e_r_direct + e_a_direct + t_direct = 1.0
        e_r_diffuse + e_a_diffuse + t_diffuse = 1.0
    
    -------------
    Computes coefficients representing the fractions of extinguished
    radiation that are absorbed, transmitted, and reflection
    
    This is the largest computational bottleneck, especially,
    the MultipleMLPs
    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):

        super(ScatteringTiming, self).__init__()
        self.n_channel = n_channel

        # DO NOT CHANGE. The MultipleMLPs assumes this value
        self.n_scattering_nets = 8  

        # Has additional input for zenith angle ('mu_direct')
        self.direct_scattering = tn.MultipleMLPs(n_input=n_constituent + 1,
                                          n_hidden_layers=3,
                                          dropout_p=dropout_p,
                                          device=device,
                                          bias=True,
                                          requires_grad=False)

        self.diffuse_scattering = tn.MultipleMLPs(n_input=n_constituent,
                                           n_hidden_layers=3,
                                           dropout_p=dropout_p,
                                           device=device,
                                           bias=True,
                                           requires_grad=False)

        # Select linear combo of 8 3x1 basis vectors to give a,r,t
        self.direct_selection = nn.Conv2d(in_channels=self.n_channel,
                                          out_channels=self.n_channel,
                                          kernel_size=(
                                              self.n_scattering_nets, 1),
                                          stride=(1, 1), padding=0, dilation=1,
                                          groups=self.n_channel, bias=False, device=device)

        self.diffuse_selection = nn.Conv2d(in_channels=self.n_channel,
                                           out_channels=self.n_channel,
                                           kernel_size=(
                                               self.n_scattering_nets, 1),
                                           stride=(1, 1), padding=0, dilation=1,
                                           groups=self.n_channel, bias=False, device=device)

        if False:
            # Remaining computation is non-essential
            # Compute number of weights
            n_hidden = [32, 32, 32]
            n_weights = n_constituent * n_hidden[0] + n_hidden[0]*n_hidden[1]
            n_weights += n_hidden[1]*n_hidden[2] + \
                n_hidden[2]*3*self.n_scattering_nets
            n_weights += n_hidden[0] + n_hidden[1] + \
                n_hidden[2] + 3*self.n_scattering_nets
            print(
                f"Scattering potential shared weights (diffuse scattering) = {n_weights}")
            print(
                f"Scattering potential shared weights (direct scattering) = {n_weights + n_hidden[0]}")
            n_weights = n_weights * 2 + n_hidden[0]
            n_weights_2 = self.n_scattering_nets * self.n_channel * 2
            print(
                f"Scattering channel specific weights  = {n_weights_2}")
            n_weights += n_weights_2
            print(f"Scattering total number of potential weights = {n_weights}")

            n_weights = n_constituent * n_hidden[0] + 64 * 4 + 64 * 4
            n_weights += 12 * 8
            n_weights += n_hidden[0] + n_hidden[1] + \
                n_hidden[2] + 3*self.n_scattering_nets
            print(
                f"Scattering actual shared weights (diffuse scattering) = {n_weights}")
            print(
                f"Scattering actual shared weights (direct scattering) = {n_weights + n_hidden[0]}")
            n_weights = n_weights * 2 + n_hidden[0]
            n_weights += self.n_scattering_nets * self.n_channel * 2
            print(f"Scattering total number of actual weights = {n_weights}")

    def reset_dropout(self, dropout_p):
        self.direct_scattering.reset_dropout(dropout_p)
        self.diffuse_scattering.reset_dropout(dropout_p)
        
    def reconfigure(self, device):
        self.direct_scattering.reconfigure(device=device)
        self.diffuse_scattering.reconfigure(device=device)

    def forward(self, x):
        (tau, mu_direct, mu_diffuse,) = x


        # tau [n_examples, n_channels, n_constituents]

        # Sum over constituents
        # Full sky (as opposed to clear sky)
        tau_full_total = torch.sum(tau, dim=2, keepdims=False)

        # Clear sky
        # tau_clear_total = torch.sum(tau[:,:,2:], dim=2, keepdims=False)

        # Direct transmission coefficients
        # Account for solar zenith angle
        # Avoids division by zero
        eps_1 = 0.0000001
        t_full_direct = torch.exp(-tau_full_total / (mu_direct + eps_1))
        t_full_diffuse = torch.exp(-tau_full_total / (mu_diffuse + eps_1))
        # t_clear = torch.exp(-tau_clear_total / (mu_direct + eps_1))

        ###### Direct Radiation ###################

        # add dimension (to be dimensionally compatible with tau)
        mu_direct = torch.unsqueeze(mu_direct, dim=2)
        tau_full_direct = tau / (mu_direct + eps_1)

        mu_direct = mu_direct.repeat(1, self.n_channel, 1)

        # Create input tensor for neural network
        # n_features = number of constituents + 1
        # where additional feature is for mu_direct
        # full_direct [n_examples,n_channels,n_features]
        full_direct = torch.concat((tau_full_direct, mu_direct), dim=2)

        # m = number of "scattering modules" that each produce
        # a basis vector
        # Each scattering module has 3 outputs
        # e_split_full_direct [n_examples,n_channels, 3 * m]
        e_split_full_direct = self.direct_scattering(full_direct)


        n = e_split_full_direct.shape[0]
        # e_split_full_direct [n_examples,n_channels, m, 3]
        e_split_full_direct = torch.reshape(
            e_split_full_direct,
            (n, self.n_channel, self.n_scattering_nets, 3))

        # Transforms into non-negative physical quantities
        e_split_full_direct = F.softmax(e_split_full_direct, dim=-1)

        # e_split_full_direct [n_examples,n_channels, 1, 3]

        e_split_full_direct = self.direct_selection(e_split_full_direct)


        # e_split_full_direct [n_examples,n_channels,3]
        e_split_full_direct = torch.squeeze(e_split_full_direct, dim=-2)

        # Each set of coefficients normalized to sum to 1.0
        # Ensures conservation of energy and non-negative physical quantities
        e_split_full_direct = F.softmax(e_split_full_direct, dim=-1)

        ###### Diffuse Radiation ###################

        # n_features = number of constituents
        # tau[n_examples,n_channels,n_features]

        # e_split_full_diffuse[n_examples,n_channels, m * 3]
        e_split_full_diffuse = self.diffuse_scattering(tau)
        n = e_split_full_diffuse.shape[0]

        # e_split_full_diffuse[n_examples,n_channels, m, 3]
        e_split_full_diffuse = torch.reshape(
            e_split_full_diffuse,
            (n, self.n_channel, self.n_scattering_nets, 3))

        # Transforms into non-negative physical quantities
        e_split_full_diffuse = F.softmax(e_split_full_diffuse, dim=-1)
        
        # e_split_full_diffuse[n_examples,n_channels, 1, 3]
        e_split_full_diffuse = self.diffuse_selection(e_split_full_diffuse)

        # e_split_full_diffuse[n_examples,n_channels, 3]
        e_split_full_diffuse = torch.squeeze(e_split_full_diffuse, dim=-2)

        # Each set of coefficients normalized to sum to 1.0
        # Ensures conservation of energy and non-negative physical quantities
        e_split_full_diffuse = F.softmax(e_split_full_diffuse, dim=-1)
        
        # Simplifies computation in MultiReflectionTiming by making the 
        # following normalization:
        #       e_t_direct + e_r_direct + e_a_direct + t_direct = 1.0
        #       e_t_diffuse + e_r_diffuse + e_a_diffuse + t_diffuse = 1.0
        
        e_t_direct = (1.0 - t_full_direct) * e_split_full_direct[:,:,0]
        e_r_direct = (1.0 - t_full_direct) * e_split_full_direct[:,:,1]
        e_a_direct = (1.0 - t_full_direct) * e_split_full_direct[:,:,2]
        
        e_t_diffuse = (1.0 - t_full_diffuse) * e_split_full_diffuse[:,:,0]
        e_r_diffuse = (1.0 - t_full_diffuse) * e_split_full_diffuse[:,:,1]
        e_a_diffuse = (1.0 - t_full_diffuse) * e_split_full_diffuse[:,:,2]

        layers = [t_full_direct, t_full_diffuse, 
                  e_t_direct, e_r_direct, e_a_direct,
                  e_t_diffuse, e_r_diffuse, e_a_diffuse]

        return layers


class MultiReflectionTiming(nn.Module):
    """ 
    Same as MultiReflection in training_network except contains
    some internal timing code, and assumes inputs are normalized
    such that 
        e_t_direct + e_r_direct + e_a_direct + t_direct = 1.0
        e_t_diffuse + e_r_diffuse + e_a_diffuse + t_diffuse = 1.0
        
    Whereas the version in training_network.py assumes:
        e_r_direct + e_a_direct + t_direct = 1.0
        e_r_diffuse + e_a_diffuse + t_diffuse = 1.0
        
    ----------------------------------------------------
    Computes each layer's "multi-reflection coefficients" by accounting
    for multireflection with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self, device):
        super(MultiReflectionTiming, self).__init__()
        self.device = device
        
    def _adding_doubling(self,
                         t_direct, t_diffuse,
                         e_t_direct, e_r_direct, e_a_direct,
                         e_t_diffuse, e_r_diffuse, e_a_diffuse,
                         r_surface_direct, r_surface_diffuse,
                         a_surface_direct, a_surface_diffuse):
        """
        Multireflection between a single layer and a (virtual) surface 
        using the adding-doubling method.

        See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
        by Grant W. Petty
        Also see Shonk and Hogan, 2007

        Input and Output Shape:
            (n_samples, n_channels, . . .)

        Arguments:

            t_direct, t_diffuse - Direct transmission coefficients of 
                the layer.  
                - These are not changed by multi reflection
                - Note that t_diffuse is for diffuse radiation that is 
                directly transmitted (not further scattered by the layer)

            e_split_direct, e_split_diffuse - The layer's split of extinguised  
                radiation into transmitted, reflected,
                and absorbed fractional components. These components 
                sum to 1.0. The transmitted and reflected components are
                used to compute downwelling and upwelling diffuse 
                radiative flux, respectively.

            r_surface_direct, r_surface_diffuse - The original reflection 
                coefficients of the (virtual) surface in isolation.

            a_surface_direct, a_surface_diffuse - The original absorption 
                coefficients of the (virtual) surface in isolation. 

        Returns:

            t_multi_direct, t_multi_diffuse - The layer's transmission
                coefficients after accounting for multi-reflection with
                the (virtual) surface immediately below it

            r_layer_multi_direct, r_layer_multi_diffuse - The layer's 
                reflection coefficients after accounting for multi-reflection 
                with the (virtual) surface immediately below it

            r_surface_multi_direct, r_surface_multi_diffuse - The (virtual)
                surface's reflection coefficients after accounting for 
                multi-reflection with the layer immediately above it

            a_layer_multi_direct, a_layer_multi_diffuse - The layer's 
                absorption coefficients layer after accounting for 
                multi-reflection with the (virtual) surface immediately
                below it

            a_surface_multi_direct, a_surface_multi_diffuse - The virtual
                surface's absorption coefficients after accounting for 
                multi-reflection with the layer immediately above it

        Notes:

            Conservation of energy:

                1.0 = a_surface_direct + r_surface_direct
                1.0 = a_surface_diffuse + r_surface_diffuse

                1.0 = a_surface_multi_direct + a_layer_multi_direct + 
                        r_layer_multi_direct
                1.0 = a_surface_multi_diffuse + a_layer_multi_diffuse + 
                        r_layer_multi_diffuse

                The absorption at the layer (after accounting for 
                multi-reflection) must equal the combined loss of flux for 
                the downward and upward streams:

                a_layer_multi_direct = (1 - t_direct - t_multi_direct) + 
                                (r_surface_multi_direct - r_layer_multi_direct)
                a_layer_multi_diffuse = (1 - t_diffuse - t_multi_diffuse) + 
                            (r_surface_multi_diffuse - r_layer_multi_diffuse)

            When merging a (virtual) surface and a layer into 
            a new virtual surface, the new surface's reflection 
            coefficient is just the reflection
            of the layer. However, the absorption of the new surface
            is the sum of the surface and layer absorptions:

                r_layer_multi_direct => r_surface_direct
                a_layer_multi_direct + a_surface_multi_direct => 
                                                            a_surface_direct

            Propagation, defined below, uses these 
            multi-reflection coefficients to propagate radiation 
            downward from the top of the atmosphere
        """

        torch.cuda.synchronize()
        t_0 = time.time()
        
        # Fractions of extinguished radiation split into transmitted, 
        # reflected, and absorbed coefficients

        eps = 1.0e-06
        
        torch.cuda.synchronize()
        t_d0 = time.time()
        
        d = 1.0/(1.0 - e_r_diffuse*r_surface_diffuse + eps)
        
        torch.cuda.synchronize()
        t_d1 = time.time()
        global t_division
        t_division += t_d1 - t_d0

        # Adding-Doubling for direct radiation
        
        t_multi_direct = (t_direct * r_surface_direct *  e_r_diffuse
                          +  e_t_direct) / (1.0 - e_r_diffuse*r_surface_diffuse + eps)
        
        torch.cuda.synchronize()
        t_d2 = time.time()
        global t_first_operation
        t_first_operation += t_d2 - t_d0
        
        r_surface_multi_direct = (t_direct * r_surface_direct * d
                                  + e_t_direct * r_surface_diffuse * d)

        a_layer_multi_direct = (e_a_direct
                                + r_surface_multi_direct * e_a_diffuse)

        r_layer_multi_direct = (e_r_direct
                                + r_surface_multi_direct
                                * (t_diffuse + e_t_diffuse))
        
        torch.cuda.synchronize()
        t_a0 = time.time()
        a_surface_multi_direct = (t_direct * a_surface_direct
                                  + t_multi_direct * a_surface_diffuse)
        torch.cuda.synchronize()
        t_a1 = time.time()
        global t_surface
        t_surface += t_a1 - t_a0

        # Adding-Doubling for diffuse radiation
        t_multi_diffuse = (
            t_diffuse * r_surface_diffuse * e_r_diffuse * d
            + e_t_diffuse * d)

        a_surface_multi_diffuse = (t_diffuse * a_surface_diffuse
                                   + t_multi_diffuse * a_surface_diffuse)

        r_surface_multi_diffuse = (t_diffuse * r_surface_diffuse * d
                                   + e_t_diffuse * r_surface_diffuse * d)

        a_layer_multi_diffuse = (e_a_diffuse
                                 + r_surface_multi_diffuse * e_a_diffuse)

        r_layer_multi_diffuse = (e_r_diffuse
                                 + r_surface_multi_diffuse
                                 * (t_diffuse + e_t_diffuse))
        
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_adding_doubling
        t_adding_doubling += t_1 - t_0

        return (t_multi_direct, t_multi_diffuse,
                r_layer_multi_direct, r_layer_multi_diffuse,
                r_surface_multi_direct, r_surface_multi_diffuse,
                a_layer_multi_direct, a_layer_multi_diffuse,
                a_surface_multi_direct, a_surface_multi_diffuse)

    def forward(self, x):
        """
        Traverses the atmospheric layers from the surface to the 
        top of the atmosphere. At each layer computes "multi-reflection"  
        coefficients modeling the effects of inter-reflection among
        the layers.

        The algorithm begins by computing the inter-reflection between 
        the surface and the nearest layer. It then
        merges this surface and layer into a new "virtual suface."
        The next iteration repeats this process with the virtual surface
        and the next nearest layer. The iterations continue until reaching 
        the top of the atmosphere (toa).

        Computations are independent across channel.

        The prefixes -- t, e, r, a -- correspond respectively to
        transmission, extinction, reflection, and absorption.
        """

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_t_direct, e_r_direct, e_a_direct, \
                  e_t_diffuse, e_r_diffuse, e_a_diffuse = radiative_layers

        # Reflection and absorption coefficients at surface
        # Add dimension for channels (but with length of 1)
        r_surface = x_surface[:, 1:2]
        a_surface = 1.0 - x_surface[:, 1:2]
        (r_surface_direct, r_surface_diffuse,
         a_surface_direct, a_surface_diffuse) = (r_surface,
                                                 r_surface,
                                                 a_surface,
                                                 a_surface)
        t_multi_direct_list = []
        t_multi_diffuse_list = []
        r_surface_multi_direct_list = []
        r_surface_multi_diffuse_list = []
        a_layer_multi_direct_list = []
        a_layer_multi_diffuse_list = []

        # Start at the original surface and move up
        # one atmospheric layer for each iteration
        for i in reversed(torch.arange(start=0, end=t_direct.shape[1])):
            # compute multi-reflection coefficients
            

            multireflected_info = self._adding_doubling(t_direct[:, i, :],
                                                        t_diffuse[:, i, :],
                                                        e_t_direct[:, i, :], 
                                                        e_r_direct[:, i, :], 
                                                        e_a_direct[:, i, :],
                                                        e_t_diffuse[:, i, :], 
                                                        e_r_diffuse[:, i, :], 
                                                        e_a_diffuse[:, i, :],
                                                        r_surface_direct,
                                                        r_surface_diffuse,
                                                        a_surface_direct,
                                                        a_surface_diffuse)

            (t_multi_direct, t_multi_diffuse,
             r_layer_multi_direct, r_layer_multi_diffuse,
             r_surface_multi_direct, r_surface_multi_diffuse,
             a_layer_multi_direct, a_layer_multi_diffuse,
             a_surface_multi_direct, a_surface_multi_diffuse) = multireflected_info

            # Merge the layer and (virtual) surface forming a new 
            # virtual surface
            r_surface_direct = r_layer_multi_direct
            r_surface_diffuse = r_layer_multi_diffuse
            a_surface_direct = a_layer_multi_direct + a_surface_multi_direct
            a_surface_diffuse = a_layer_multi_diffuse + a_surface_multi_diffuse

            t_multi_direct_list.append(t_multi_direct)
            t_multi_diffuse_list.append(t_multi_diffuse)
            r_surface_multi_direct_list.append(r_surface_multi_direct)
            r_surface_multi_diffuse_list.append(r_surface_multi_diffuse)
            a_layer_multi_direct_list.append(a_layer_multi_direct)
            a_layer_multi_diffuse_list.append(a_layer_multi_diffuse)

        # Stack output in layers
        t_multi_direct = torch.stack(t_multi_direct_list, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse_list, dim=1)
        r_surface_multi_direct = torch.stack(
            r_surface_multi_direct_list, dim=1)
        r_surface_multi_diffuse = torch.stack(
            r_surface_multi_diffuse_list, dim=1)
        a_layer_multi_direct = torch.stack(a_layer_multi_direct_list, dim=1)
        a_layer_multi_diffuse = torch.stack(a_layer_multi_diffuse_list, dim=1)

        # Reverse ordering of layers such that top layer is first
        t_multi_direct = torch.flip(t_multi_direct, dims=(1,))
        t_multi_diffuse = torch.flip(t_multi_diffuse, dims=(1,))
        r_surface_multi_direct = torch.flip(r_surface_multi_direct, dims=(1,))
        r_surface_multi_diffuse = torch.flip(
            r_surface_multi_diffuse, dims=(1,))
        a_layer_multi_direct = torch.flip(a_layer_multi_direct, dims=(1,))
        a_layer_multi_diffuse = torch.flip(a_layer_multi_diffuse, dims=(1,))

        multireflected_layers = [t_direct, t_diffuse,
                                 t_multi_direct, t_multi_diffuse,
                                 r_surface_multi_direct, r_surface_multi_diffuse,
                                 a_layer_multi_direct, a_layer_multi_diffuse]
        
        # The reflection coefficient at the top of the atmosphere
        # is the reflection coefficient of top layer
        upward_reflection_toa = r_layer_multi_direct
        return (multireflected_layers, upward_reflection_toa)



class FullNetTiming(nn.Module):
    """ 
    Same as FullNet in train_network.py except contains
    timing code and slightly different output for ScatteringTiming
    and input for MultiReflectionTiming
    
    Propagates flux from the top of the atmosphere to the
    surface, making a single pass through the atmospheric layers

    Values for flux_direct, flux_diffuse propagate into each layer for
    each channel: 
                
    Downward Direct Flux Transmitted = flux_direct * t_direct
    Downward Diffuse Flux Transmitted = 
                    flux_direct * t_multi_direct + 
                    flux_diffuse * (t_diffuse + t_multi_diffuse)

    Upward Flux from Top Layer = flux_direct * r_layer_multi_direct +
                            flux_diffuse * r_layer_multi_diffuse

    Upward Flux into Top Layer from below = 
                        flux_direct * r_surface_multi_direct +
                        flux_diffuse * r_surface_multi_diffuse

    Upward fluxes are diffuse only because they are due to scattering
    of the downward fluxes

    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):
        super(FullNetTiming, self).__init__()
        with torch.no_grad():
            self.device = device
            self.n_channel = n_channel
            self.solar_constant = 1361.0

            # Learns single diffuse zenith angle approximation
            self.mu_diffuse_net = nn.Linear(1, 1, bias=False, device=device)
            torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)

            # Learns decompositon of input solar radiation into channels
            self.spectral_net = nn.Linear(1, n_channel, bias=False, device=device)
            torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)

            # Learns parameters for computing optical depth for each 
            # constituent for each channel. Same for all atmospheric layers
            self.optical_depth_net = tn.LayerDistributed(tn.OpticalDepth(n_channel,
                                                                dropout_p,
                                                                device))
            # Learns parameters for computing scattering for each channel.
            # Same for all atmospheric layers 
            
            if tn.__name__ == 'train_network':
                self.scattering_net = tn.LayerDistributed(ScatteringTiming(n_channel,
                                                            n_constituent,
                                                            dropout_p,
                                                            device))
                # Combines optical properties of atmospheric layers
                # accounting for inter-reflection.
                # Uses adding-doubling algorith. Math only. No learning.
                self.multireflection_net = MultiReflectionTiming(device)
            else:
                self.scattering_net = tn.LayerDistributed(tn.Scattering(n_channel,
                                                            n_constituent,
                                                            dropout_p,
                                                            device))
                # Combines optical properties of atmospheric layers
                # accounting for inter-reflection.
                # Uses adding-doubling algorith. Math only. No learning.
                self.multireflection_net = tn.MultiReflection()
                
            # Propagates radiation from top of atmosphere (TOA) to surface
            # Math only. No learning.
            self.propagation_net = tn.Propagation(n_channel)

    def reset_dropout(self, dropout_p):
        self.optical_depth_net.reset_dropout(dropout_p)
        self.scattering_net.reset_dropout(dropout_p)

    def forward(self, x):

        x_layers, x_surface, _, _, _, = x

        # inputs for each layer: temp, pressure, masses of atmospheric 
        # constituents
        (temperature_pressure, constituent_mass) = (x_layers[:, :, 0:2],
                                                    x_layers[:, :, 2:10])

        mu_direct = x_surface[:, 0]

        # Dummy input of ones to network for diffuse effective zenith angle
        one = torch.ones((1,), dtype=torch.float32,
                         device=self.device)

        mu_diffuse = torch.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse.reshape((-1, 1, 1))

        # (1, n_layers, 1)
        mu_diffuse = mu_diffuse.repeat(
            [x_layers.shape[0], x_layers.shape[1], 1])

        mu_direct_layered = mu_direct.reshape((-1, 1, 1))
        
        # (n_examples, n_layers, 1)
        mu_direct_layered = mu_direct_layered.repeat([1, x_layers.shape[1], 1])

        # Compute optical depth across examples, layers, channels and
        # atmospheric constituents
        
        # (n_examples, n_layers, n_channels, n_constituents)
        torch.cuda.synchronize()
        t_0 = time.time()
        tau = self.optical_depth_net((temperature_pressure,
                                      constituent_mass))
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_optical_depth
        t_optical_depth += t_1 - t_0
        
        # Compute scattering coefficients
        torch.cuda.synchronize()
        t_0 = time.time()
        layers = self.scattering_net((tau, mu_direct_layered, mu_diffuse,))
        #print(f"Requires Grad = {layers[0].requires_grad}")
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_scattering
        t_scattering += t_1 - t_0
        
        # Compute multilayer coefficients
        torch.cuda.synchronize()
        t_0 = time.time()
        (multireflected_layers,
         upward_reflection_toa) = self.multireflection_net((layers, x_surface,))
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_multireflection
        t_multireflection += t_1 - t_0
        
        # Compute split of incoming direct flux
        flux_direct = F.softmax(self.spectral_net(
            one), dim=-1) * self.solar_constant
        
        # Account for solar zenith angle
        flux_direct = torch.unsqueeze(
            flux_direct, dim=0) * mu_direct.reshape((-1, 1))

        # set incoming diffuse flux to zero
        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                   dtype=torch.float32,
                                   device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        # Propagate radiation from top of atmosphere to surface
        torch.cuda.synchronize()
        t_0 = time.time()
        flux = self.propagation_net((multireflected_layers,
                                    upward_reflection_toa,
                                    input_flux))
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_propagation
        t_propagation += t_1 - t_0
        
        (flux_down_direct, flux_down_diffuse, flux_up_diffuse,
         flux_absorbed) = flux

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]

class FullNetInternals(nn.Module):
    """ 
    Same as FullNet in train_network.py except
    that it also outputs various internal variables
    
    Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column
    
    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):
        super(FullNetInternals, self).__init__()
        self.device = device
        self.n_channel = n_channel
        self.solar_constant = 1361.0

        # Learns single diffuse zenith angle approximation
        self.mu_diffuse_net = nn.Linear(1, 1, bias=False, device=device)
        torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)

        # Learns decompositon of input solar radiation into channels
        self.spectral_net = nn.Linear(1, n_channel, bias=False, device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)

        # Learns optical depth for each layer for each constituent for
        # each channel
        self.optical_depth_net = tn.LayerDistributed(tn.OpticalDepth(n_channel, dropout_p,
                                                                     device))

        self.scattering_net = tn.LayerDistributed(tn.Scattering(n_channel,
                                                                n_constituent,
                                                                dropout_p,
                                                                device))

        self.multireflection_net = tn.MultiReflection()

        # Propagates radiation from top of atmosphere (TOA) to surface
        self.propagation_net = tn.Propagation(n_channel)


    def forward(self, x):
        x_layers, x_surface, _, _, _, = x

        (temperature_pressure, constituent_mass) = (x_layers[:, :, 0:2],
                                                    x_layers[:, :, 2:10])

        mu_direct = x_surface[:, 0]

        one = torch.ones((1,), dtype=torch.float32,
                         device=self.device)

        mu_diffuse_original = torch.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse_original.reshape((-1, 1, 1))
        # mu_diffuse = torch.unsqueeze(mu_diffuse,dim=(1,2))
        # (1, n_layers, 1)
        mu_diffuse = mu_diffuse.repeat(
            [x_layers.shape[0], x_layers.shape[1], 1])

        # mu_direct = torch.unsqueeze(mu_direct,dim=(1,2))
        mu_direct_layered = mu_direct.reshape((-1, 1, 1))
        # (n_examples, n_layers, 1)
        mu_direct_layered = mu_direct_layered.repeat([1, x_layers.shape[1], 1])

        # (n_examples, n_layers, n_channels, n_constituents)
        tau = self.optical_depth_net((temperature_pressure, constituent_mass))

        layers = self.scattering_net((tau, mu_direct_layered, mu_diffuse,))

        # extinguished layers[i,layers,channels,3]
        t_direct, t_diffuse, e_split_direct, e_split_diffuse = layers
        
        # scattering fraction per channel
        s_direct_channels = (
            1.0 - t_direct) * (e_split_direct[:, :, :, 0] + e_split_direct[:, :, :, 1])
        s_diffuse_channels = (
            1.0 - t_diffuse) * (e_split_diffuse[:, :, :, 0] + e_split_diffuse[:, :, :, 1])
        #

        (multireflected_layers,
         upward_reflection_toa) = self.multireflection_net((layers, x_surface,))

        channel_split = F.softmax(self.spectral_net(one), dim=-1)

        r_toa = upward_reflection_toa * channel_split.reshape((1, -1))
        # sum over channels
        r_toa = torch.sum(r_toa, dim=1, keepdim=False)
        #

        flux_direct = torch.unsqueeze(
            channel_split, dim=0) * mu_direct.reshape((-1, 1)) * self.solar_constant

        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                   dtype=torch.float32,
                                   device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        flux = self.propagation_net((multireflected_layers,
                                    upward_reflection_toa,
                                    input_flux))

        channel_split = channel_split.reshape((1, 1, -1))
        # Weight the channels appropriately
        # (i, 1, n_channels)
        s_direct_channels = s_direct_channels * channel_split
        s_diffuse_channels = s_diffuse_channels * channel_split

        t_direct_total = t_direct * channel_split
        t_diffuse_total = t_diffuse * channel_split
        
        # Scattering fraction for entire layer
        s_direct = torch.sum(s_direct_channels, dim=2, keepdim=False)
        s_diffuse = torch.sum(s_diffuse_channels, dim=2, keepdim=False)
        t_direct_total = torch.sum(t_direct_total, dim=2, keepdim=False)
        t_diffuse_total = torch.sum(t_diffuse_total, dim=2, keepdim=False)

        (flux_down_direct, flux_down_diffuse, flux_up_diffuse,
         flux_absorbed) = flux

        internal_data = [x_layers[:, :, 2], x_layers[:, :, 3], x_layers[:, :, 5], mu_diffuse_original, s_direct, s_diffuse, r_toa, x_surface[:, 1],
                         mu_direct, t_direct_total, t_diffuse_total, x_layers[:, :, 4]]
        predicted_data = [flux_down_direct,
                          flux_down_diffuse, flux_up_diffuse, flux_absorbed]

        return predicted_data, internal_data

def test_layers_loop(dataloader, model, loss_functions, loss_names, loss_weights, is_flux, device):
    """ 
    Generic testing / evaluation loop 
    Computes an error metric for each layer
    """
    model.eval()
    num_batches = len(dataloader)

    # Determining number of layers
    dataset = dataloader.dataset
    sample, _, _, _, _ = dataset[0]
    sample_shape = sample.shape

    # Loss for each atmospheric column
    if is_flux:
        loss = np.zeros(
            (len(loss_functions), sample_shape[0] + 1), dtype=np.float32)
    else:
        loss = np.zeros(
            (len(loss_functions), sample_shape[0]), dtype=np.float32)
    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i, :] += loss_fn(data, y_pred, loss_weights).cpu().numpy()

    loss /= num_batches

    print(f"Test Error: ")
    for i, values in enumerate(loss):
        print(f" {loss_names[i]}:")
        for j, value in enumerate(values):
            print(f"   {j}. {value:.4f}")
    print("")

    return loss

# computes an error metric for each geographic location


def test_geographic_loop(dataloader, model, loss_functions, loss_names, loss_weights, number_of_sites, loss_file_name, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    # Loss for each geographic location
    loss = np.zeros((len(loss_functions), number_of_sites), dtype=np.float32)

    count = np.zeros((len(loss_functions), number_of_sites), dtype=np.int32)

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                tmp_loss, tmp_count = loss_fn(data, y_pred, loss_weights)
                loss[i, :] += tmp_loss.numpy()
                count[i, :] += tmp_count.numpy()

    for i, name in enumerate(loss_names):
        loss[i, :] = loss[i, :] / np.float32(count[i, :])
        if name.find("rmse") > 0:
            print(f"Computing RMSE for {name}")
            loss[i, :] = np.sqrt(loss[i, :])
        else:
            print(f"Computing bias for {name}")

    dt = Dataset(loss_file_name, "w")
    dim1 = dt.createDimension("sites", number_of_sites)
    for i, name in enumerate(loss_names):
        var = dt.createVariable(name, "f4", ("sites",))
        var[:] = loss[i, :]

    dt.close()
    return loss


def test_loop_internals(dataloader, model, loss_functions, loss_names, loss_weights, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    lwp = []
    iwp = []
    o3 = []
    mu_diffuse = []
    s_direct = []
    s_diffuse = []
    r_toa = []
    r_surface = []
    mu_direct = []
    t_direct = []
    t_diffuse = []
    h2o = []
    # squared_loss = []

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred, internal_data = model(data)
            lwp.append(internal_data[0])
            iwp.append(internal_data[1])
            o3.append(internal_data[2])
            mu_diffuse.append(internal_data[3])
            s_direct.append(internal_data[4])
            s_diffuse.append(internal_data[5])
            r_toa.append(internal_data[6])
            r_surface.append(internal_data[7])
            mu_direct.append(internal_data[8])
            t_direct.append(internal_data[9])
            t_diffuse.append(internal_data[10])
            h2o.append(internal_data[11])

            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, loss_weights).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.4f}")
    print("")

    lwp = torch.cat(lwp, dim=0)
    iwp = torch.cat(iwp, dim=0)
    o3 = torch.cat(o3, dim=0)
    mu_diffuse = torch.cat(mu_diffuse, dim=0)
    mu_direct = torch.cat(mu_direct, dim=0)
    s_direct = torch.cat(s_direct, dim=0)
    s_diffuse = torch.cat(s_diffuse, dim=0)
    r_toa = torch.cat(r_toa, dim=0)
    r_surface = torch.cat(r_surface, dim=0)
    t_direct = torch.cat(t_direct, dim=0)
    t_diffuse = torch.cat(t_diffuse, dim=0)
    h2o = torch.cat(h2o, dim=0)

    internal_data = [lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse,
                     r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o]

    return loss, internal_data


def write_internal_data(internal_data, output_file_name):
    import xarray as xr
    lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse, r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o = internal_data

    shape = lwp.shape

    example = np.arange(shape[0])
    layer = np.arange(shape[1])

    # lwp = xr.DataArray(lwp, coords=[time,site,layer], dims=("time","site","layer"), name="lwp")

    # iwp = xr.DataArray(iwp, coords=[time,site,layer], dims=("time","site","layer"), name="iwp")

    # r = xr.DataArray(r, coords=[time,site,layer],dims=("time","site","layer"), name="r")

    mu_diffuse = mu_diffuse.cpu().numpy().flatten()

    mu_diffuse_n = np.arange(mu_diffuse.shape[0])
    mu_direct = mu_direct.cpu().numpy()
    # s1 = np.shape(mu_direct)
    # mu_direct = np.reshape(mu_direct, (s1[0], s1[1]*s1[2]))

    rs_direct = s_direct.cpu().numpy()
    rs_diffuse = s_diffuse.cpu().numpy()
    rr_toa = r_toa.cpu().numpy()
    rr_surface = r_surface.cpu().numpy()

    is_bad = np.isnan(rs_direct).any() or np.isnan(rs_diffuse).any()
    print(f"is bad = {is_bad}")

    ds = xr.Dataset(
        data_vars={
            "lwp": (["example", "layer"], lwp.cpu().numpy()),
            "iwp": (["example", "layer"], iwp.cpu().numpy()),
            "o3": (["example", "layer"], o3.cpu().numpy()),
            "mu_diffuse": (["mu_diffuse_n"], mu_diffuse),
            "mu_direct": (["example"], mu_direct),
            "s_direct": (["example", "layer"], rs_direct),
            "s_diffuse": (["example", "layer"], rs_diffuse),
            "r_toa": (["example"], rr_toa),
            "r_surface": (["example"], rr_surface),
            "t_direct": (["example", "layer"], t_direct.cpu().numpy()),
            "t_diffuse": (["example", "layer"], t_diffuse.cpu().numpy()),
            "h2o": (["example", "layer"], h2o.cpu().numpy()),
        },
        coords={
            "example": example,
            "layer": layer,
            "mu_diffuse_n": mu_diffuse_n,
        },
    )

    ds.to_netcdf(output_file_name)
    ds.close()

def test_loop_timing(dataloader, model, loss_functions, loss_names, loss_weights, device):
    """ Generic testing / evaluation loop """

    model.eval()

    t_delt = 0.0

    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    with torch.no_grad():
    #with torch.inference_mode():
        for data in dataloader:
            data = [x.to(device) for x in data]
            torch.cuda.synchronize()
            t_0 = time.time()
            y_pred = model(data)
            torch.cuda.synchronize()
            t_1 = time.time()
            t_delt += t_1 - t_0
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, loss_weights).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.4f}")
    print(f"Time Evaluation (w/o data I/O) = {t_delt} s")
    global t_optical_depth
    global t_scattering
    global t_multireflection
    global t_propagation
    print(f"Time optical depth = {t_optical_depth} s")
    print(f"Time scattering = {t_scattering} s") 
    print(f"Time multireflection = {t_multireflection} s") 
    print(f"Time propagation = {t_propagation} s") 
    t_optical_depth = 0.0
    t_scattering = 0.0
    t_multireflection = 0.0
    t_propagation = 0.0
    global t_adding_doubling
    global t_division
    global t_first_operation
    global t_surface
    if tn.__name__ == 'train_network':
        print(f"Time adding_doubling = {t_adding_doubling} s") 
        print(f"Time division = {t_division} s") 
        print(f"Time first operation = {t_first_operation} s") 
        print(f"Time surface = {t_surface} s") 
    t_adding_doubling = 0.0
    t_division = 0.0
    t_first_operation = 0.0
    t_surface = 0.0
    print("")

    return loss

def evaluate_network_analysis():
    """
    Allows a trained network to analyzed in various ways:
    - Computation time used by each module
    - Error as a function of geography
    - Error as a function of atmospheric layer
    - Output of internal network variables (e.g. learned cos zenith angle)
    - Error as a function of solar zenith angle
    - Error on "clear sky" (clouds removed) data
    
    Also can be used to analyze various versions of trained network
    - train_network_3 - Allows each gas to influence each channel
    - train_network_4 - Alternative inputs to scattering module
    - train_network_5 - 28 channels
    - train_network_6 - 14 channels
    - train_network_7 - Replacing transmissivity and scattering modules with single
      neural network modules for direct and diffuse radiation
    """
    
    # Specify model
    base_dir = "/home/hws/src/openbox_neural_networks/shortwave_radiative_transfer/"
    model_dir = base_dir + "models/"
    model_name_prefix = 'openbox.shortwave.'
    # To change models, change the package inported as "tn" 
    if tn.__name__ == 'train_network':
        model_id = "v2."                # "Baseline" trained model
        n_epoch = 596                   # Epoch selected by training
        n_channel = 42
    elif tn.__name__ == 'train_network_3':
        model_id = "v3.1."            
        n_epoch = 752
        n_channel = 42
    elif tn.__name__ == 'train_network_4':
        model_id = "v4.1."            
        n_epoch = 753
        n_channel = 42
    elif tn.__name__ == 'train_network_5':
        model_id = "v5.1."            
        n_epoch = 669
        n_channel = 28
    elif tn.__name__ == 'train_network_6':
        model_id = "v6.2."            
        n_epoch = 655
        n_channel = 14
    elif tn.__name__ == 'train_network_7':
        model_id = "v7.1."            
        n_epoch = 765
        n_channel = 42


    # Specify dataset
    if True:
        mode = "testing"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/testing/"
        years = ("2009", "2015", "2020")
    else:
        mode = "training" #"validation"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/training/"
        years = ("2008", )

    # Analysis Options:
    
    # At most one of the following hould be set True:
    # is_use_internals, is_geographic_loss, is_layered_loss,
    # is_clear_sky
    # If none are set to True, computes standard losses averaged 
    # over the entire dataset

    # For writing internal variable values, e.g., learned cosine zenith angle,
    # to netcdf file
    is_use_internals = False
    geographic_loss_file_name = base_dir + f"{model_name_prefix}{model_id}.geographic_error."

    # Computes losses wrt to geographic location
    is_geographic_loss = False

    # Computes loss of each atmospheric layer individually
    is_layered_loss = False

    # When is_layered_loss=True, only one error (heating rate, up flux, 
    # down flux) may be computed at a time 
    is_flux = False  # only matters when is_layered_loss = True or is_geographic_loss
    is_down = False  # only matters when is_layered_loss = True or is_geographic_loss

    # Computes losses for clear sky data
    is_clear_sky = False
    
    # Computational speed for OpticalDepth may be reduced by
    # adding @torch.compile decorator to its forward() method
    is_timing = True
    
    print(" ")
    print(f"Model = {tn.__name__}")
    print(f"Timing = {is_timing}")
    print(f"Write layer by layer loss = {is_layered_loss}")
    if is_layered_loss:
        if is_flux:
            if is_down:
                print(" for downwelling flux")
            else:
                print(" for upwelling flux")
        else:
                print("  for the heating rate")
    print(f"Write network's internal values = {is_use_internals}")
    print(f"Write loss as a function of geographic location = {is_geographic_loss}")
    print(f"Clear sky data = {is_clear_sky}")
    print("")
    
    if is_layered_loss and is_geographic_loss:
        print(f"Cannot generated layered loss and geographic loss simultaneously.")
        print(f"Set either is_layered_loss or is_geographic_loss to False")
        quit()
        
    if is_layered_loss and is_use_internals:
        print(f"Cannot generated layered loss and generate internal data simultaneously.")
        print(f"Set either is_layered_loss or is_use_internals to False")
        quit()
        
    if is_geographic_loss and is_use_internals:
        print(f"Cannot generate geographic loss and generate internal data simultaneously.")
        print(f"Set either is_geographic_loss or is_use_internals to False")
        quit()
    
    if is_layered_loss or is_geographic_loss or is_use_internals:
        if tn.__name__ != "train_network":
            print(f"Not implemented for {tn.__name__}")
            quit()
    elif is_timing:
        if tn.__name__ == 'train_network_4':
            print("Not implemented.")
            print(f"FullNetTiming cannot currently accommodate {tn.__name__}")
            print("OpticalDepth transmits different info to Scattering")
            quit()
        if tn.__name__ == 'train_network_7':
            print("Not implemented.")
            print(f"FullNetTiming cannot currently accommodate {tn.__name__}")
            print(f"{tn.__name__} uses a OpticalProperties module in place of")
            print("Transmissivity and Scattering modules")
            quit()
            
    if False:
        # Hardcode to cpu
        print("Pytorch version:", torch.__version__)
        device = "cpu"
        print(f"Using {device} device")

    else:
        print("Pytorch version:", torch.__version__)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} device")

        if device == "cuda":
            print('__CUDNN VERSION:', torch.backends.cudnn.version())
            print('__Number CUDA Devices:', torch.cuda.device_count())
            print('__CUDA Device Name:', torch.cuda.get_device_name(0))
            print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(
                0).total_memory/1e9)
            print(f'Device capability = {torch.cuda.get_device_capability()}')

    batch_size = 4096               # Reduce if program runs out of memory
                                    
    n_constituent = 8               # Constant. Do not change

    # Ideally, this should be read from data file, rather than being
    # hardcoded
    number_of_sites = 5120

    # Setup model
    if is_use_internals:
        model = FullNetInternals(
            n_channel, n_constituent, dropout_p=0, device=device)
    elif is_timing:
        model = FullNetTiming(n_channel, n_constituent,
                           dropout_p=0, device=device)
    else:
        model = tn.FullNet(n_channel, n_constituent,
                           dropout_p=0, device=device)

    model = model.to(device=device)
    model_filename = model_dir + f"{model_name_prefix}{model_id}"
    
    loss_weights = tn.get_loss_weights(n_epoch)
    checkpoint = torch.load(model_filename + str(n_epoch).zfill(3),
                            map_location=torch.device(device))
    print(f"Loaded Model: epoch = {n_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Cycle through testing sets
    for year in years:
        test_input_dir = f"{processed_data_dir}{year}/"
        months = [str(m).zfill(2) for m in range(1, 13)]

        test_input_files = [
            f'{test_input_dir}shortwave-{mode}-{year}-{month}.nc'
            for month in months]

        print(f"Loading {mode} dataset for {year}")
        if is_clear_sky:
            print("Using clear sky data")
        test_dataset = data_generation.RTDataSet(test_input_files,
                                                 is_clear_sky=is_clear_sky)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size,
                                                      shuffle=False,
                                                      num_workers=1)
        if is_geographic_loss:

            geographic_heating_rate_rmse = nl.geographic_heating_rate_maker(
                nl.geographic_rmse, number_of_sites)
            geographic_heating_rate_bias = nl.geographic_heating_rate_maker(
                nl.geographic_bias, number_of_sites)

            geographic_down_flux_rmse = nl.geographic_flux_maker(
                nl.geographic_rmse, number_of_sites, is_down=True)
            geographic_down_flux_bias = nl.geographic_flux_maker(
                nl.geographic_bias, number_of_sites, is_down=True)

            geographic_up_flux_rmse = nl.geographic_flux_maker(
                nl.geographic_rmse, number_of_sites, is_down=False)
            geographic_up_flux_bias = nl.geographic_flux_maker(
                nl.geographic_bias, number_of_sites, is_down=False)

            geographic_loss_functions = (
                geographic_heating_rate_rmse,
                geographic_heating_rate_bias,
                geographic_down_flux_rmse,
                geographic_down_flux_bias,
                geographic_up_flux_rmse,
                geographic_up_flux_bias)

            geographic_loss_names = ("heating_rate_rmse", 
                                     "heating_rate_bias",
                                     "downwelling_flux_rmse",
                                     "downwelling_flux_bias", 
                                     "upwelling_flux_rmse", 
                                     "upwelling_flux_bias")
            _ = test_geographic_loop(
                test_dataloader, model, geographic_loss_functions,
                geographic_loss_names, loss_weights, number_of_sites,
                geographic_loss_file_name + str(n_epoch).zfill(3) + f".{year}.nc", device)

        elif is_layered_loss:
            if not is_flux:
                # Compute heating rate losses

                layered_loss_functions = (nl.layered_heating_rate_rmse,
                                          nl.layered_heating_rate_bias)
            else:
                if is_down:
                    layered_loss_functions = (
                        nl.layered_downwelling_flux_rmse, 
                        nl.layered_downwelling_flux_bias)
                else:
                    layered_loss_functions = (
                        nl.layered_upwelling_flux_rmse, 
                        nl.layered_upwelling_flux_bias)

            if is_flux:
                if is_down:
                    layered_loss_names = (
                        "downwelling flux rmse", "downwelling flux bias")
                else:
                    layered_loss_names = (
                        "upwelling flux rmse", "upwelling flux bias")
            else:
                layered_loss_names = (
                    "heating rate rmse", "heating rate bias")

            _ = test_layers_loop(
                test_dataloader, model, layered_loss_functions,
                layered_loss_names, loss_weights, is_flux, device)
        else:
            loss_flux_0_01 = nl.mu_selector_flux_maker(0.01, nl.total_rmse)
            loss_flux_0_05 = nl.mu_selector_flux_maker(0.05, nl.total_rmse)
            loss_flux_0_10 = nl.mu_selector_flux_maker(0.10, nl.total_rmse)

            # Standard loss functions
            loss_functions = (
                nl.openbox_rmse,
                nl.flux_rmse, nl.heating_rate_rmse, 
                nl.direct_flux_rmse, nl.diffuse_flux_rmse,
                nl.flux_bias, nl.downwelling_flux_rmse, nl.upwelling_flux_rmse,
                nl.downwelling_flux_bias, nl.upwelling_flux_bias,
                nl.direct_extinction_rmse,
                nl.diffuse_heating_rate_rmse,
                nl.heating_rate_bias,
                loss_flux_0_01, loss_flux_0_05, loss_flux_0_10)

            loss_names = (
                "Openbox RMSE",
                "Flux RMSE", "Heating Rate RMSE", 
                "Direct Flux RMSE", "Diffuse Flux RMSE",
                "Flux Bias", "Flux Down RMSE", "Flux up RMSE",
                "Flux Down Bias", "Flux up Bias ",
                "Direct Extinction RMSE",
                "Diffuse Heating Rate RMSE",
                "Heating Rate Bias",
                "RMSE_flux_0_01", "RMSE_flux_0_05", "RMSE_flux_0_10")

            if is_use_internals:
                _, internal_data = test_loop_internals(
                    test_dataloader, model, loss_functions, loss_names,
                    loss_weights, device)
                write_internal_data(
                    internal_data, output_file_name=test_input_dir +
                    f"internal_output.sc_{model_id}_{n_epoch}.{year}.nc")

            elif not is_layered_loss and not is_geographic_loss:
                if is_timing:
                    torch.cuda.synchronize()
                    t_0 = time.time()
                    _ = test_loop_timing(test_dataloader, model,
                                    loss_functions,
                                    loss_names,
                                    loss_weights, device)
                    global t_total
                    torch.cuda.synchronize()
                    t_1 = time.time()
                    t_delta = t_1 - t_0
                    t_total += t_delta
                    print(f"Evaluation time (including data I/O) = {t_delta} s")
                    print(" ")
                else:
                    _ = tn.test_loop(test_dataloader, model,
                                    loss_functions,
                                    loss_names,
                                    loss_weights, device)
    if is_timing:
        print(f"Total evaluation time= {t_total} s")


def evaluate_network():
    """
    Evaluates accuracy of chosen network model on testing sets 
    from 2009, 2015, 2020
    """
    
    # Specify model location
    model_dir = "/home/hws/src/openbox_neural_networks/shortwave_radiative_transfer/models/"
    model_name_prefix = 'openbox.shortwave.'
    
    if tn.__name__ == 'train_network':  # "Baseline" trained model
        model_id = "v2."                
        n_epoch = 596                   # Epoch selected by training
        n_channel = 42
    elif tn.__name__ == 'train_network_3': 
        model_id = "v3.1."    # Allows each gas to influence each channel        
        n_epoch = 752
        n_channel = 42
    elif tn.__name__ == 'train_network_4':
        model_id = "v4.1."      # Alternative inputs to scattering module      
        n_epoch = 753
        n_channel = 42
    elif tn.__name__ == 'train_network_5':
        model_id = "v5.1."      # 28 channels instead of 42 
        n_epoch = 669
        n_channel = 28
    elif tn.__name__ == 'train_network_6':
        model_id = "v6.2."      # 14 channels instead of 42       
        n_epoch = 655
        n_channel = 14
    elif tn.__name__ == 'train_network_7':
        # Replaces transmissivity and scattering modules with single
        # neural network modules for direct and diffuse radiation
        model_id = "v7.1."            
        n_epoch = 765
        n_channel = 42
    
    # Specify datasets
    mode = "testing"
    processed_data_dir = "/data-T1/hws/CAMS/processed_data/testing/"
    years = ("2009", "2015", "2020")
    
    # Use GPU if available
    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if device == "cuda":
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(
            0).total_memory/1e9)
        print(f'Device capability = {torch.cuda.get_device_capability()}')

    batch_size = 4096               # Reduce if program runs out of memory

    # constants
    n_channel = 42                                                 
    n_constituent = 8       

    # Set up model
    model = tn.FullNet(n_channel, n_constituent,
                           dropout_p=0, device=device)
    
    model = model.to(device=device)
    
    model_filename = model_dir + f"{model_name_prefix}{model_id}"
    
    checkpoint = torch.load(model_filename + str(n_epoch).zfill(3),
                            map_location=torch.device(device))
    print(" ")
    print(f"Model = {tn.__name__}")
    print(f"Loaded Model: epoch = {n_epoch}")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Gets "loss weights" for open box loss function from "weighting schedule"
    loss_weights = tn.get_loss_weights(n_epoch)
    
    for year in years:
        test_input_dir = f"{processed_data_dir}{year}/"
        months = [str(m).zfill(2) for m in range(1, 13)]

        test_input_files = [
            f'{test_input_dir}shortwave-{mode}-{year}-{month}.nc'
            for month in months]

        print(f"Loading {mode} dataset for {year}")

        test_dataset = data_generation.RTDataSet(test_input_files,
                                                 is_clear_sky=False)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size,
                                                      shuffle=False,
                                                      num_workers=1)
        loss_functions = (
            nl.openbox_rmse,
            nl.flux_rmse, nl.heating_rate_rmse,
            nl.direct_flux_rmse, nl.diffuse_flux_rmse,
            nl.flux_bias, nl.downwelling_flux_rmse, nl.upwelling_flux_rmse,
            nl.downwelling_flux_bias, nl.upwelling_flux_bias,
            nl.direct_extinction_rmse,
            nl.diffuse_heating_rate_rmse,
            nl.heating_rate_bias)

        loss_names = (
            "Openbox RMSE", 
            "Flux RMSE", "Heating Rate RMSE", 
            "Direct Flux RMSE", "Diffuse Flux RMSE",
            "Flux Bias", "Flux Down RMSE", "Flux up RMSE",
            "Flux Down Bias", "Flux up Bias ",
            "Direct Extinction RMSE",
            "Diffuse Heating Rate RMSE",
            "Heating Rate Bias")

        _ = tn.test_loop(test_dataloader, 
                         model, 
                         loss_functions, 
                         loss_names,
                         loss_weights, 
                         device)

if __name__ == "__main__":
    
    # Choose network to evaluate by importing it as "tn", 
    # at the top of this file. For example:
    #
    # import training_network_5.py as tn
    
    # If is_analysis = False, computes accuracy of chosen network on
    # testing sets
    
    # If True, analyzes the results in various ways. See 
    # evaluate_network_analysis()
    
    is_analysis = False            

    if is_analysis:
        evaluate_network_analysis()
    else:
        evaluate_network()
