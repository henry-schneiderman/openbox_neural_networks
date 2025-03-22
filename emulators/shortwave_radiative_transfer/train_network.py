from netCDF4 import Dataset
import numpy as np
import time
from typing import List
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

import data_generation
import network_losses as nl
# Used to avoid division by zero
eps_1 = 0.0000001

class MLP(nn.Module):
    """
    Multi Layer Perceptron (MLP) module

    Fully connected layers
    Uses ReLU() activation for hidden units
    No activation for output unit
    
    Initialization of all weights with uniform distribution with 'lower' 
    and 'upper' bounds. Defaults to -0.1 < weight < 0.1
    
    Hidden units initial bias with uniform distribution 0.9 < x < 1.1
    Output unit initial bias with uniform distribution -0.1 < x <0.1
    """

    def __init__(self, n_input, n_hidden: List[int], n_output, 
                 dropout_p, device, lower=-0.1, upper=0.1, bias=True):
        super(MLP, self).__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_output
        n_last = n_input
        self.hidden = nn.ModuleList()

        for n in n_hidden:
            mod = nn.Linear(n_last, n, bias=bias, device=device)
            torch.nn.init.uniform_(mod.weight, a=lower, b=upper)
            # Bias initialized to ~1.0
            # Because of ReLU activation, don't want any connections to
            # be prematurely pruned away by becoming negative.
            # Therefore start with a significant positive bias
            if bias:
                torch.nn.init.uniform_(mod.bias, a=0.9, b=1.1) 
            self.hidden.append(mod)
            n_last = n
        self.dropout_p = dropout_p
        self.output = nn.Linear(n_last, n_output, bias=bias, device=device)
        torch.nn.init.uniform_(self.output.weight, a=lower, b=upper)
        if bias:
            torch.nn.init.uniform_(self.output.bias, a=-0.1, b=0.1)

    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p


    def forward(self, x):

        for hidden in self.hidden:
            x = hidden(x)
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
        return self.output(x)

class BD(nn.Module):
    """
    Block Diagonal (BD) module

    Computes several fully connected modules in parallel using single
    computation with large matrix multiplications, where the the 
    separate computatation are split using a block diagonal matrix.

    The initial stage between the input and first layer of hidden
    units is fully connected. All hidden layers have
    32 nodes and that output has 24 nodes.
    All block diagonals contain 4 8x8 masks except for output which 
    contains 8 4x3 masks
    """

    def __init__(
        self, n_input, n_hidden, n_output, dropout_p, device, bias=False): 

        super(BD, self).__init__()
        hidden_n = 32
        output_n = 24
        self.n_hidden = n_hidden
        self.dropout_p = dropout_p
       
        weight_values = torch.rand(
            (n_input, self.n_hidden[0]), requires_grad=True,
            device=device, dtype=torch.float32)
        
        self.input_weight = nn.parameter.Parameter(weight_values, 
                                                   requires_grad=True)

        self.bias = bias
        if bias:
            bias_values = torch.rand(
                (self.n_hidden[0],), requires_grad=True, device=device,
                dtype=torch.float32,
                )
            self.input_bias = nn.parameter.Parameter(
                bias_values, requires_grad=True)
            biases = []

        template = torch.ones((8,8), device=device, dtype=torch.float32)
        self.filter = torch.block_diag(template, template, template, template)
        weights = []

        n_last = n_hidden[0]
        for n in n_hidden[1:]:
            weights.append(
                torch.rand(
                    (n_last, n), requires_grad=True,
                    device=device, dtype=torch.float32))
            if bias:
                biases.append(
                    torch.rand(
                        (n,), requires_grad=True, 
                        device=device, dtype=torch.float32))
            n_last = n

        self.weights = torch.nn.ParameterList(weights)
        tmp_weights = torch.rand(
            (n_last, n_output), requires_grad=True, device=device,
            dtype=torch.float32)

        self.output_weights = nn.parameter.Parameter(
            tmp_weights, requires_grad=True)

        if bias:
            self.biases = torch.nn.ParameterList(biases)
            weight_values = torch.rand(
                (n_output,), requires_grad=True, device=device,
                dtype=torch.float32)

            self.output_bias = nn.parameter.Parameter(
                weight_values, requires_grad=True)

        template = torch.ones(
            (4,3), device=device, dtype=torch.float32)
        self.output_filter = torch.block_diag (
            template,template,template,template, 
            template,template,template,template)
            
    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p

    def forward(self, x):

        if self.bias:
            x = x @ self.input_weight + self.input_bias
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
            for i, weight in enumerate(self.weights):
                x = x @ (self.filter * weight) + self.biases[i]
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout_p,training=self.training)
            x = x @ (self.output_filter * self.output_weights) + self.output_bias
        else:
            x = x @ self.input_weight
            x = F.relu(x)
            x = F.dropout(x,p=self.dropout_p,training=self.training)
            for weight in self.weights:
                x = x @ (self.filter * weight)
                x = F.relu(x)
                x = F.dropout(x,p=self.dropout_p,training=self.training)
            x = x @ (self.output_filter * self.output_weights)
        return x

class LayerDistributed(nn.Module):
    """
    Applies a nn.Module independently to an array of inputs (layers)

    Same idea as TensorFlow's TimeDistributed Class
    Adapted from:
    https://stackoverflow.com/questions/62912239/tensorflows-timedistributed-equivalent-in-pytorch
    The input and output may each be a single
    tensor or a list of tensors.

    Each tensor has dimensions: (n_samples, n_layers, data's dimensions. . .)
    """

    def __init__(self, module):
        super(LayerDistributed, self).__init__()
        self.module = module

    def reset_dropout(self,dropout_p):
        self.module.reset_dropout(dropout_p)

    def forward(self, x):
        if torch.is_tensor(x):
            shape = x.shape
            n_sample = shape[0]
            n_layer = shape[1]
            # Squash samples and layers into a single dimension
            squashed_input = x.contiguous().view(n_sample*n_layer, *shape[2:]) 
        else: 
            # else 'x' is a list of tensors. Squash each individually
            squashed_input = []
            for xx in x:
                # Squash samples and layers into a single dimension
                shape = xx.shape
                n_sample = shape[0]
                n_layer = shape[1]
                xx_reshape = xx.contiguous().view(n_sample*n_layer, *shape[2:])
                squashed_input.append(xx_reshape)
        y = self.module(squashed_input)
        # Reshape y
        if torch.is_tensor(y):
            shape = y.shape
            unsquashed_output = y.contiguous().view(n_sample, n_layer, 
                                                    *shape[1:]) 
        else:
            # else 'y' is a list of tensors. Unsquash each individually
            unsquashed_output = []
            for yy in y:
                shape = yy.shape
                yy_reshaped = yy.contiguous().view(n_sample, n_layer, 
                                                   *shape[1:])
                unsquashed_output.append(yy_reshaped)
        return unsquashed_output


class Extinction(nn.Module):
    """ 
    Computes influence of each atmospheric constituent on each spectral
    channel
    Generates optical depth for each atmospheric 
    constituent for each channel for the given layer.
    
    Learns the dependence of mass extinction coefficients on temperature 
    and pressure.
    Hard-codes the multiplication of each mass
    extinction coefficient by the consistuent's mass

    Consider using better model for liquid water and ice water (cloud
    optics). See (Hogan and Bozzo, 2018)

    Inputs:
        Mass of each atmospheric constituent
        Temperature, pressure

    Outputs
        Optical depth of each constituent in each channel
    """
    def __init__(self, n_channel, dropout_p, device):
        super(Extinction, self).__init__()
        self.n_channel = n_channel
        self.device = device
        self.dropout_p = dropout_p

        # Computes a scalar weight for each constituent 
        # for each channel
        self.net_lw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_iw  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_h2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o3  = nn.Linear(1,self.n_channel,bias=False,device=device)

        self.net_co2 = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_o2  = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_n2o = nn.Linear(1,self.n_channel,bias=False,device=device)
        self.net_ch4 = nn.Linear(1,self.n_channel,bias=False,device=device)
        #self.net_co = nn.Linear(1,self.n_channel,bias=False,device=device)

        n_weights = 8 * n_channel

        print(f'number of possible weights gas channel decomposition = {n_weights}')

        lower = -0.9 # exp(-0.9) = .406
        upper = 0.5  # exp(0.5) = 1.64
        torch.nn.init.uniform_(self.net_lw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_iw.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_h2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o3.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_co2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_o2.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_n2o.weight, a=lower, b=upper)
        torch.nn.init.uniform_(self.net_ch4.weight, a=lower, b=upper)
        #torch.nn.init.uniform_(self.net_co.weight, a=lower, b=upper)

        # exp() activation forces weight to be non-negative

        # Modifies each extinction coeffient as a function of temperature, 
        # pressure 
        # Seeks to model pressuring broadening of atmospheric absorption lines
        # Single network for each constituent
        self.net_ke_h2o = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                            dropout_p=dropout_p,device=device)
        self.net_ke_o3  = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_co2 = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_o2   = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_n2o = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        self.net_ke_ch4 = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
                              dropout_p=dropout_p,device=device)
        #self.net_ke_co = MLP(n_input=2,n_hidden=(6,4,4),n_output=1,
        #                      dropout_p=dropout_p,device=device)

        n_weights_ext = 6 * (12 + 24 + 16 + 4 + 6 + 4 + 4 + 1)
        print(f"Extinction n weights = {n_weights_ext} with 6 atmospheric constituents")

        n_weights_ext = 8 * (12 + 24 + 16 + 4 + 6 + 4 + 4 + 1)
        print(f"Extinction n weights = {n_weights_ext} with 8 atmospheric constituents")


        # Filters select which channels each constituent contributes to
        # Follows similiar assignment of bands as
        # Table A2 in Pincus, R., Mlawer, E. J., &
        # Delamere, J. S. (2019). Balancing accuracy, efficiency, and 
        # flexibility in radiation calculations for dynamical models. Journal 
        # of Advances in Modeling Earth Systems, 11,3074–3089. 
        # https://doi.org/10.1029/2019MS001621


        filter_h2o = torch.tensor([1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,],
                                       dtype=torch.float32,device=device)
        
        self.filter_h2o = torch.cat([filter_h2o, filter_h2o, filter_h2o])

        filter_o3 = torch.tensor([1,0,0,0,0, 0,0,0,1,1, 1,0,1,1,],
                                       dtype=torch.float32,device=device)

        self.filter_o3 = torch.cat([filter_o3, filter_o3, filter_o3])

        filter_co2 = torch.tensor([1,0,1,0,1, 0,1,0,0,0, 0,0,0,0,],
                                       dtype=torch.float32,device=device)
        
        self.filter_co2 = torch.cat([filter_co2, filter_co2, filter_co2])
        
        #filter_u  = torch.tensor([1,0,0,0,0, 0,0,1,1,1, 1,0,0,1],
        filter_o2  = torch.tensor([1,0,0,0,0, 0,0,1,1,1, 1,1,0,1],
                                       dtype=torch.float32,device=device)
        
        self.filter_o2 = torch.cat([filter_o2, filter_o2, filter_o2])

        filter_n2o = torch.tensor([1,0,0,0,0, 0,0,0,0,0, 0,0,0,0],
                                       dtype=torch.float32,device=device)

        self.filter_n2o = torch.cat([filter_n2o, filter_n2o, filter_n2o])
        
        filter_ch4 = torch.tensor([1,1,0,1,0, 1,0,0,0,0, 0,0,0,0,],
                                       dtype=torch.float32,device=device)
        
        self.filter_ch4 = torch.cat([filter_ch4, filter_ch4, filter_ch4])

        n_weights_decomposition = (2 * n_channel + torch.sum(self.filter_ch4)
                     + torch.sum(self.filter_o3) + torch.sum(self.filter_co2)
                     + torch.sum(self.filter_o2) + torch.sum(self.filter_n2o)
                     + torch.sum(self.filter_h2o)) # + torch.sum(self.filter_co))

        print(f"Gas decomposition actual trainable weights = {n_weights_decomposition}")

    def reset_dropout(self,dropout_p):
        self.dropout_p = dropout_p
        self.net_ke_h2o.reset_dropout(dropout_p)
        self.net_ke_o3.reset_dropout(dropout_p)
        self.net_ke_co2.reset_dropout(dropout_p)
        self.net_ke_o2.reset_dropout(dropout_p)
        self.net_ke_n2o.reset_dropout(dropout_p)
        self.net_ke_ch4.reset_dropout(dropout_p)
        #self.net_ke_co.reset_dropout(dropout_p)

    def forward(self, x):

        temperature_pressure, constituents = x

        c = constituents
        shape = c.shape
        c = c.reshape((shape[0],1,shape[1]))
        t_p = temperature_pressure

        # Activation functions        
        a = torch.exp
        b = torch.sigmoid

        # Dummy input to compute weights of atmospheric constituents
        one = torch.ones((shape[0],1),dtype=torch.float32,device=self.device)
        # a(self.net_lw (one)): (n_examples, n_channels)
        # tau_lw : (n_examples, n_channels) * (n_examples, 1)
        tau_lw  = a(self.net_lw (one)) * (c[:,:,0])
        tau_iw  = a(self.net_iw (one)) * (c[:,:,1])
        # tau_h2o : (n_examples, n_channels) * (n_examples, 1) *
        # ((n_channels) * (n_examples, 1))
        tau_h2o = a(self.net_h2o(one)) * (c[:,:,2]) * (self.filter_h2o * 
                                                       b(self.net_ke_h2o(t_p)))
        tau_o3  = a(self.net_o3 (one)) * (c[:,:,3] * (self.filter_o3  * 
                                                       b(self.net_ke_o3 (t_p))))
        tau_co2 = a(self.net_co2(one)) * (c[:,:,4]) * (self.filter_co2 * 
                                                       b(self.net_ke_co2(t_p)))
        tau_o2   = a(self.net_o2  (one)) * (c[:,:,5]) * (self.filter_o2 * 
                                                       b(self.net_ke_o2  (t_p)))
        tau_n2o = a(self.net_n2o(one)) * (c[:,:,6]) * (self.filter_n2o * 
                                                       b(self.net_ke_n2o(t_p)))
        tau_ch4 = a(self.net_ch4(one)) * (c[:,:,7]) * (self.filter_ch4 * 
                                                       b(self.net_ke_ch4(t_p)))

        #tau_co = a(self.net_co(one)) * (c[:,:,8]) * (self.filter_co * 
        #                                               b(self.net_ke_co(t_p)))


        tau_lw  = torch.unsqueeze(tau_lw,2)
        tau_iw  = torch.unsqueeze(tau_iw,2)
        tau_h2o = torch.unsqueeze(tau_h2o,2)
        tau_o3  = torch.unsqueeze(tau_o3,2)
        tau_co2 = torch.unsqueeze(tau_co2,2)

        tau_o2   = torch.unsqueeze(tau_o2,2)
        tau_n2o = torch.unsqueeze(tau_n2o,2)
        tau_ch4 = torch.unsqueeze(tau_ch4,2)
        #tau_co = torch.unsqueeze(tau_co,2)

        tau = torch.cat([tau_lw, tau_iw, tau_h2o, tau_o3, tau_co2, tau_o2, 
                         tau_n2o, tau_ch4],dim=2)

        return tau


class Scattering_v2_tau_efficient(nn.Module):
    """ 
    Computes split of each coefficient of extinguished radiation into 
    three coefficients represented the fractions that are absorbed, 
    transmitted, and reflected


    """

    def __init__(self, n_channel, n_constituent, dropout_p, device):

        super(Scattering_v2_tau_efficient, self).__init__()
        self.n_channel = n_channel
        #self.n_scattering_nets = 5 # settings for 8
        self.n_scattering_nets = 8 # settings for 11

        n_input = n_constituent #8

        #tmp_array = np.ones((n_constituent), dtype=np.float32)
        #tmp_array[0] = 0.0
        #tmp_array[1] = 0.0
        #tmp_array = tmp_array.reshape((1,1,-1))
        #self.clear_sky_mask = tensorize(tmp_array).to(device)

        #n_hidden = [7, 7, 7, 7]  # settings for 8


        #n_hidden = [24, 24, 24, 24] # settings for 12
        n_hidden = [32, 32, 32] # settings for 13

        # Create basis functions for scattering

        # Has additional input for zenith angle ('mu_direct')

        self.direct_scattering = BD(n_input=n_input + 1,
                                        n_hidden=n_hidden,
                                        n_output=24,
                                        dropout_p=dropout_p,
                                        device=device,
                                        bias=True) 


        self.diffuse_scattering = BD(n_input=n_input, 
                                    n_hidden=n_hidden, 
                                    n_output=24,
                                    dropout_p=dropout_p,
                                    device=device,
                                    bias=True) 

        # Select combo of basis to give a,r,t

        self.direct_selection = nn.Conv2d(in_channels=self.n_channel,
                                          out_channels=self.n_channel, 
                                          kernel_size=(self.n_scattering_nets, 1), 
                                          stride=(1,1), padding=0, dilation=1, 
                                          groups=self.n_channel, bias=False, device=device)

        self.diffuse_selection = nn.Conv2d(in_channels=self.n_channel,
                                          out_channels=self.n_channel, 
                                          kernel_size=(self.n_scattering_nets,1), 
                                          stride=(1,1), padding=0, dilation=1, 
                                          groups=self.n_channel, bias=False, device=device)

        n_weights = n_input * n_hidden[0] + n_hidden[0]*n_hidden[1]
        n_weights += n_hidden[1]*n_hidden[2] + n_hidden[2]*3*self.n_scattering_nets
        n_weights += n_hidden[0] + n_hidden[1] + n_hidden[2] + 3*self.n_scattering_nets
        print (f"Number of potential shared weights diffuse scattering = {n_weights}")
        print (f"Number of potential shared weights direct scattering = {n_weights + n_hidden[0]}")
        n_weights = n_weights * 2 + n_hidden[0]
        n_weights_2 = self.n_scattering_nets * self.n_channel * 2
        print(f"Number of channel specific weights scattering  = {n_weights_2}")
        n_weights += n_weights_2
        print(f"Total number of potential scattering weights = {n_weights}")

        n_weights = n_input * n_hidden[0] + 64 * 4 + 64 * 4 
        n_weights += 12 * 8 
        n_weights += n_hidden[0] + n_hidden[1] + n_hidden[2] + 3*self.n_scattering_nets
        print (f"Number of actual shared weights diffuse scattering = {n_weights}")
        print (f"Number of actual shared weights direct scattering = {n_weights + n_hidden[0]}")
        n_weights = n_weights * 2 + n_hidden[0]
        n_weights += self.n_scattering_nets * self.n_channel * 2
        print(f"Total number of actual learned weights = {n_weights}")

    def reset_dropout(self,dropout_p):
        self.direct_scattering.reset_dropout(dropout_p)
        self.diffuse_scattering.reset_dropout(dropout_p)


    def forward(self, x):
        (tau, mu_direct, mu_diffuse,) = x

        # sum over constituents
        # n_examples, n_channels, n_constituents

        # Full sky
        tau_full_total = torch.sum(tau, dim=2, keepdims=False)

        # Clear sky
        #tau_clear_total = torch.sum(tau[:,:,2:], dim=2, keepdims=False)


        t_full_direct = torch.exp(-tau_full_total / (mu_direct + eps_1))
        t_full_diffuse = torch.exp(-tau_full_total / (mu_diffuse + eps_1))
        #t_clear = torch.exp(-tau_clear_total / (mu_direct + eps_1))

        ###### Process Diffuse Radiation ###################

        # add dimension for constituents
        mu_direct = torch.unsqueeze(mu_direct,dim=2)
        tau_full_direct = tau / (mu_direct + eps_1)

        mu_direct = mu_direct.repeat(1,self.n_channel,1)

        full_direct = torch.concat((tau_full_direct, mu_direct), dim=2)
        # f = number of features = number of constituents + 1
        # [i,channels,f]
        e_split_full_direct = self.direct_scattering(full_direct)

        # m = number of scattering nets
        # [i,channels, 3 * m]
        n = e_split_full_direct.shape[0]
        e_split_full_direct = torch.reshape(
            e_split_full_direct, 
            (n, self.n_channel, self.n_scattering_nets, 3))

        # [i,channels, m, 3]
        e_split_full_direct = F.softmax(e_split_full_direct,dim=-1)

        e_split_full_direct = self.direct_selection(e_split_full_direct)

        # [i, channels, 1, 3]

        e_split_full_direct = torch.squeeze(e_split_full_direct, dim=-2)

        # [i,channels,3]  

        e_split_full_direct = F.softmax(e_split_full_direct, dim=-1)

        ###### Process Diffuse Radiation ###################

        # f = number of features = number of constituents
        # [i,channels,f]

        e_split_full_diffuse = self.diffuse_scattering(tau)
        n = e_split_full_diffuse.shape[0]
        
        # [i,channels,3, m]

        e_split_full_diffuse = torch.reshape(
            e_split_full_diffuse,
            (n, self.n_channel, self.n_scattering_nets, 3))

        # [i,channels, m, 3]
        e_split_full_diffuse = F.softmax(e_split_full_diffuse,dim=-1)

        e_split_full_diffuse = self.diffuse_selection(e_split_full_diffuse)

        # [i, channels, 1, 3]

        e_split_full_diffuse = torch.squeeze(e_split_full_diffuse, dim=-2)

        # [i,channels,3]  

        e_split_full_diffuse = F.softmax(e_split_full_diffuse, dim=-1)

 

        layers = [t_full_direct, t_full_diffuse, e_split_full_direct, 
                  e_split_full_diffuse] 

        return layers


class MultiReflection(nn.Module):
    """ 
    Computes each layer's "multi-reflection coefficients" by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self):
        super(MultiReflection, self).__init__()

    def _adding_doubling (self, 
                          t_direct, t_diffuse, 
                          e_split_direct, e_split_diffuse, 
                          r_surface_direct, r_surface_diffuse, 
                          a_surface_direct, a_surface_diffuse):

        """
        Multireflection between a single layer and a (virtual) surface 
        using the Adding-Doubling Method.

        See p.418-424 of "A First Course in Atmospheric Radiation (2nd edition)"
        by Grant W. Petty
        Also see Shonk and Hogan, 2007

        Input and Output Shape:
            (n_samples, n_channels, . . .)

        Arguments:

            t_direct, t_diffuse - Direct transmission coefficients of 
                the layer.  
                - These are not changed by multi reflection
                - t_diffuse is for diffuse input that is directly 
                transmitted.

            e_split_direct, e_split_diffuse - The layer's split of extinguised  
                radiation into transmitted, reflected,
                and absorbed components. These components 
                sum to 1.0. The transmitted and reflected components produce
                diffuse radiation.
                
            r_surface_direct, r_surface_diffuse - The original reflection 
                coefficients of the surface.

            a_surface_direct, a_surface_diffuse - The original absorption 
                coefficients of the surface. 
                
        Returns:

            t_multi_direct, t_multi_diffuse - The layer's transmission
                coefficients for radiation that is multi-reflected (as 
                opposed to directly transmitted, e.g., t_direct, t_diffuse)

            r_layer_multi_direct, r_layer_multi_diffuse - The layer's 
                reflection coefficients after accounting for multi-reflection 
                with the surface

            r_surface_multi_direct, r_surface_multi_diffuse - The surface's
                reflection coefficients after accounting for 
                multi-reflection with the layer

            a_layer_multi_direct, a_layer_multi_diffuse - The layer's 
                absorption coefficients layer after accounting for 
                multi-reflection with surface

            a_surface_multi_direct, a_surface_multi_diffuse - The surface's
                absorption coefficients after accounting for multi-reflection 
                with the layer

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

            When merging the multireflected layer and the surface into 
            a new "surface", the reflection coefficient is just the reflection
            of the layer. However, the absorption of the new surface
            is the sum of the surface and layer absorptions:

                r_layer_multi_direct => r_surface_direct
                a_layer_multi_direct + a_surface_multi_direct => 
                                                            a_surface_direct

            The Propagation class below uses the multi-reflection
            coefficients to propagate radiation 
            downward from the top of the atmosphere
        """
        # Split out extinguished component
        e_direct = 1.0 - t_direct
        e_diffuse = 1.0 - t_diffuse

        # Split extinguished into transmitted, reflected, and absorbed
        e_t_direct, e_r_direct, e_a_direct = (e_split_direct[:,:,0], 
                                              e_split_direct[:,:,1],
                                              e_split_direct[:,:,2])
        e_t_diffuse, e_r_diffuse, e_a_diffuse = (e_split_diffuse[:,:,0], 
                                                 e_split_diffuse[:,:,1],
                                                 e_split_diffuse[:,:,2])

        eps = 1.0e-06
        d = 1.0/(1.0 - e_diffuse*e_r_diffuse*r_surface_diffuse + eps)

        # Adding-Doubling for direct radiation
        t_multi_direct = (t_direct* r_surface_direct * e_diffuse * e_r_diffuse*d 
                        + e_direct * e_t_direct * d)
        
        a_surface_multi_direct = (t_direct * a_surface_direct 
                                + t_multi_direct * a_surface_diffuse)

        r_surface_multi_direct = (t_direct * r_surface_direct * d 
                                + e_direct * e_t_direct * r_surface_diffuse * d)

        a_layer_multi_direct = (e_direct * e_a_direct 
                            + r_surface_multi_direct * e_diffuse * e_a_diffuse)

        r_layer_multi_direct = (e_direct * e_r_direct 
                        + r_surface_multi_direct 
                        * (t_diffuse + e_diffuse * e_t_diffuse))

        # Adding-Doubling for diffuse radiation
        t_multi_diffuse = (
            t_diffuse * r_surface_diffuse * e_diffuse * e_r_diffuse * d 
            + e_diffuse * e_t_diffuse * d) 
        
        a_surface_multi_diffuse = (t_diffuse * a_surface_diffuse 
                                + t_multi_diffuse * a_surface_diffuse)

        r_surface_multi_diffuse = (t_diffuse * r_surface_diffuse * d 
                             + e_diffuse * e_t_diffuse * r_surface_diffuse*d)
        
        a_layer_multi_diffuse = (e_diffuse * e_a_diffuse 
                        + r_surface_multi_diffuse * e_diffuse * e_a_diffuse)

        r_layer_multi_diffuse = (e_diffuse * e_r_diffuse 
                        + r_surface_multi_diffuse 
                        * (t_diffuse + e_diffuse * e_t_diffuse))

        return (t_multi_direct, t_multi_diffuse, 
                r_layer_multi_direct, r_layer_multi_diffuse, 
                r_surface_multi_direct, r_surface_multi_diffuse, 
                a_layer_multi_direct, a_layer_multi_diffuse, 
                a_surface_multi_direct, a_surface_multi_diffuse)

    def forward(self, x):
        """
        Traverses the atmospheric layer from the surface to the 
        top of the atmosphere. At each layer generates "multi-reflection"  
        coefficients modeling the effects of inter-reflection among
        the layers.

        The algorithm begins by computing the inter-reflection between 
        the surface and the nearest surface. It then
        merges this surface and layer into a new "virtual suface."
        The next iteration repeats this process with the virtual surface
        the next layer. The iterations continue until reaching 
        the top of the atmosphere (toa).

        Computations are independent across channel.

        The prefixes -- t, e, r, a -- correspond respectively to
        transmission, extinction, reflection, and absorption.
        """

        radiative_layers, x_surface = x

        t_direct, t_diffuse, e_split_direct, e_split_diffuse = radiative_layers

        # Add dimension for channels (but with length of 1)
        r_surface = x_surface[:,1:2]
        a_surface = 1.0 - x_surface[:,1:2]
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

        # Start at the original surface and the first layer and move up
        # one atmospheric layer for each iteration
        for i in reversed(torch.arange(start=0, end=t_direct.shape[1])):
            # compute multi-reflection coefficients
            multireflected_info = self._adding_doubling (t_direct[:,i,:], 
                                                   t_diffuse[:,i,:], 
                                                   e_split_direct[:,i,:,:], 
                                                   e_split_diffuse[:,i,:,:], 
                                                   r_surface_direct, 
                                                   r_surface_diffuse, 
                                                   a_surface_direct, 
                                                   a_surface_diffuse)
            (t_multi_direct, t_multi_diffuse,
            r_layer_multi_direct, r_layer_multi_diffuse,
            r_surface_multi_direct, r_surface_multi_diffuse,
            a_layer_multi_direct, a_layer_multi_diffuse,
            a_surface_multi_direct, a_surface_multi_diffuse) = multireflected_info

            # Merge the layer and surface forming a new "virtual surface"
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
        t_multi_direct= torch.stack(t_multi_direct_list, dim=1)
        t_multi_diffuse = torch.stack(t_multi_diffuse_list, dim=1)
        r_surface_multi_direct = torch.stack(r_surface_multi_direct_list, dim=1)
        r_surface_multi_diffuse = torch.stack(r_surface_multi_diffuse_list, dim=1)
        a_layer_multi_direct = torch.stack(a_layer_multi_direct_list, dim=1)
        a_layer_multi_diffuse = torch.stack(a_layer_multi_diffuse_list, dim=1)

        # Reverse ordering of layers such that top layer is first
        t_multi_direct = torch.flip(t_multi_direct, dims=(1,))
        t_multi_diffuse = torch.flip(t_multi_diffuse, dims=(1,))
        r_surface_multi_direct = torch.flip(r_surface_multi_direct, dims=(1,))
        r_surface_multi_diffuse = torch.flip(r_surface_multi_diffuse, dims=(1,))
        a_layer_multi_direct = torch.flip(a_layer_multi_direct, dims=(1,))
        a_layer_multi_diffuse = torch.flip(a_layer_multi_diffuse, dims=(1,))

        multireflected_layers = [t_direct, t_diffuse, 
                                 t_multi_direct, t_multi_diffuse, 
                                 r_surface_multi_direct,r_surface_multi_diffuse, 
                                 a_layer_multi_direct, a_layer_multi_diffuse]
        # The reflection coefficient at the top of the atmosphere
        # is the reflection coefficient of top layer
        upward_reflection_toa = r_layer_multi_direct
        return (multireflected_layers, upward_reflection_toa)

    
class Propagation(nn.Module):
    """
    Propagate flux from the top of the atmosphere to the
    surface.
    Makes a single pass through the atmosphere

    Consider two downward fluxes entering the layer: 
                flux_direct, flux_diffuse

    Downward Direct Flux Transmitted = flux_direct * t_direct
    Downward Diffuse Flux Transmitted = 
                    flux_direct * t_multi_direct + 
                    flux_diffuse * (t_diffuse + t_multi_diffuse)

    Upward Flux from Top Layer = flux_direct * r_layer_multi_direct +
                            flux_diffuse * r_layer_multi_diffuse

    Upward Flux into Top Layer from below = 
                        flux_direct * r_surface_multi_direct +
                        flux_diffuse * r_surface_multi_diffuse

    Upward fluxes are diffuse since they are from radiation
    that is scattered upwards
    """
    def __init__(self,n_channel):
        super().__init__()
        super(Propagation, self).__init__()
        self.n_channel = n_channel

    def forward(self, x):

        multireflected_layers, upward_reflection_toa, input_flux = x

        (t_direct, t_diffuse,
        t_multi_direct, t_multi_diffuse,
        r_surface_multi_direct, r_surface_multi_diffuse,
        a_layer_multi_direct, a_layer_multi_diffuse)  = multireflected_layers

        flux_direct, flux_diffuse = input_flux

        # Assign all 3 fluxes above the top layer
        flux_down_direct = [flux_direct]
        flux_down_diffuse = [flux_diffuse]
        flux_up_diffuse = [flux_direct * upward_reflection_toa]

        flux_absorbed = []

        # Propagate downward through the atmospheric layers
        for i in range(t_direct.shape[1]):

            flux_absorbed.append(flux_direct * a_layer_multi_direct[:,i] + flux_diffuse * a_layer_multi_diffuse[:,i])

            # Will want this later when incorporate surface interactions:
            #flux_absorbed_surface = flux_direct * a_surface_multi_direct + \
            #flux_diffuse * a_surface_multi_diffuse

            flux_down_direct.append(flux_direct * t_direct[:,i])
            flux_down_diffuse.append(
                flux_direct * t_multi_direct[:,i] 
                + flux_diffuse * (t_diffuse[:,i] + t_multi_diffuse[:,i]))
            
            flux_up_diffuse.append(
                flux_direct * r_surface_multi_direct[:,i] 
                + flux_diffuse * r_surface_multi_diffuse[:,i])
            
            flux_direct = flux_down_direct[-1]
            flux_diffuse = flux_down_diffuse[-1]
        
        # stack atmospheric layers
        flux_down_direct = torch.stack(flux_down_direct,dim=1)
        flux_down_diffuse = torch.stack(flux_down_diffuse,dim=1)
        flux_up_diffuse = torch.stack(flux_up_diffuse,dim=1)
        flux_absorbed = torch.stack(flux_absorbed,dim=1)
  
        # Sum across channels
        flux_down_direct = torch.sum(flux_down_direct,dim=2,keepdim=False)
        flux_down_diffuse = torch.sum(flux_down_diffuse,dim=2,keepdim=False)      
        flux_up_diffuse = torch.sum(flux_up_diffuse,dim=2,keepdim=False)  
        flux_absorbed = torch.sum(flux_absorbed,dim=2,keepdim=False)  

        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
                flux_absorbed]

class FullNet(nn.Module):
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

    def __init__(self, n_channel, n_constituent, dropout_p, device):
        super(FullNet, self).__init__()
        self.device = device
        self.n_channel = n_channel
        self.solar_constant = 1361.0

        # Learns single diffuse zenith angle approximation 
        self.mu_diffuse_net = nn.Linear(1,1,bias=False,device=device)
        torch.nn.init.uniform_(self.mu_diffuse_net.weight, a=0.4, b=0.6)

        # Learns decompositon of input solar radiation into channels
        self.spectral_net = nn.Linear(1,n_channel,bias=False,device=device)
        torch.nn.init.uniform_(self.spectral_net.weight, a=0.4, b=0.6)

        # Learns optical depth for each layer for each constituent for 
        # each channel
        self.extinction_net = LayerDistributed(Extinction(n_channel,dropout_p,
                                                          device))
        
        self.scattering_net = LayerDistributed(Scattering_v2_tau_efficient(n_channel,
                                                    n_constituent,
                                                    dropout_p,
                                                    device))

        self.multireflection_net = MultiReflection()

        # Propagates radiation from top of atmosphere (TOA) to surface
        self.propagation_net = Propagation(n_channel)

    def reset_dropout(self,dropout_p):
        self.extinction_net.reset_dropout(dropout_p)
        self.scattering_net.reset_dropout(dropout_p)

    def forward(self, x):

        x_layers, x_surface, _, _, _, = x

        (temperature_pressure, 
        constituents) = (x_layers[:,:,0:2], 
                        x_layers[:,:,2:10])
        
        mu_direct = x_surface[:,0] 

        one = torch.ones((1,),dtype=torch.float32,
                        device=self.device)
        
        mu_diffuse = torch.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse.reshape((-1,1,1))
        #mu_diffuse = torch.unsqueeze(mu_diffuse,dim=(1,2))
        #(1, n_layers, 1)
        mu_diffuse = mu_diffuse.repeat([x_layers.shape[0],x_layers.shape[1],1])

        #mu_direct = torch.unsqueeze(mu_direct,dim=(1,2))
        mu_direct_layered = mu_direct.reshape((-1,1,1))
        #(n_examples, n_layers, 1)
        mu_direct_layered = mu_direct_layered.repeat([1,x_layers.shape[1],1])

        #(n_examples, n_layers, n_channels, n_constituents)
        tau = self.extinction_net((temperature_pressure, 
                                constituents))

        layers = self.scattering_net((tau, mu_direct_layered, mu_diffuse,))

        (multireflected_layers, 
        upward_reflection_toa) = self.multireflection_net((layers,x_surface,))

        flux_direct = F.softmax(self.spectral_net(one),dim=-1) * self.solar_constant
        flux_direct = torch.unsqueeze(flux_direct,dim=0) * mu_direct.reshape((-1,1))
        #flux_direct = flux_direct.repeat([mu_direct.shape[0],1])

        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                        dtype=torch.float32,
                                        device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        flux = self.propagation_net((multireflected_layers, 
                                    upward_reflection_toa,
                                    input_flux))

        (flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
        flux_absorbed) = flux
        
        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]



def train_loop(dataloader, model, optimizer, loss_function, loss_weights, device):
    """ Generic training loop """

    model.train()

    loss_string = "Training Loss: "
    for batch, data in enumerate(dataloader):
        data = [x.to(device) for x in data]
        y_pred = model(data)

        loss = loss_function(data, y_pred, loss_weights)

        loss.backward()
        torch.cuda.synchronize()

        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"

def test_loop(dataloader, model, loss_functions, loss_names, loss_weights, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, loss_weights).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
    print("")

    return loss

def get_loss_weights(n):
    if n <= 200:
        loss_weights = [2.0, 1.0, 0.5, 0.25]
    elif n <= 285:
        loss_weights = [1.0, 1.0, 0.5, 0.5]
    elif n <= 360:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    elif n <= 515:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    elif n <= 600:
        loss_weights = [1.0, 1.0, 0.5, 0.5]
    else:
        loss_weights = [0.5, 2.0, 1.0, 1.0]
    return loss_weights

def get_dropout (n, n_epochs):
    dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) 
    dropout_epochs =   (-1, 40, 60, 70,  80, 90,  105, 120, 135, n_epochs + 1)

    dropout_index = next(i for i, epoch in enumerate(dropout_epochs) if n <= epoch) - 1
    dropout_p = dropout_schedule[dropout_index]
    return dropout_p

def train_network():

    print("Pytorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:',torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        print(f'Device capability = {torch.cuda.get_device_capability()}')
        use_cuda = True
    else:
        use_cuda = False

    # Might help reduce computational time
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32 = True

    n_initial_models = 4
    # Set epoch at which to start training. May continue from any
    # epoch for which model exists. Otherwise, set to zero
    n_start = 0
    # If n_start == 1, then n_best must be set to index
    # of best initial model
    n_best = -1 

    model_id = "v2.1."  

    batch_size = 1024

    n_epochs = 2000
    n_windup = 200
    n_stop_training = 50 
    checkpoint_period = 5

    model_dir     = "/data-T1/hws/tmp/"
    model_filename = model_dir + f"/Torch.SW.{model_id}" 

    train_input_dir = "/data-T1/hws/CAMS/processed_data/training/2008/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]
    train_input_files = [f'{train_input_dir}nn_input_sw-training-2008-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}nn_input_sw-cross_validation-2008-{month}.nc' for month in months]

    ##################################
    # DO NOT CHANGE. Assumed by OpticalDepth
    n_channel = 42
    n_constituent = 8
    ##################

    best_loss = 1.0e+08

    n_elapsed_best = 0

    train_dataset = data_generation.RTDataSet(train_input_files, is_clear_sky=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=False, num_workers=1)
    
    validation_dataset = data_generation.RTDataSet(cross_input_files, is_clear_sky=False)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size, shuffle=False, num_workers=1)

    loss_functions = (
        nl.openbox_rmse,  nl.flux_rmse, nl.direct_flux_rmse, 
        nl.diffuse_flux_rmse, nl.flux_bias, 
        nl.heating_rate_rmse, nl.direct_extinction_rmse,
        nl.diffuse_heating_rate_rmse)

    loss_names = (
        "Openbox RMSE", "Flux RMSE", "Direct Flux RMSE",
        "Diffuse Flux RMSE", "Flux Bias", 
        "Heating Rate RMSE","Direct Extinction RMSE", 
        "Diffuse Heating Rate RMSE")

    dropout_p = 0.0
    last_dropout_p = 0.0
    if n_start == 0:
        # Start by training several "initial models" for the first 
        # epoch. Select the best one

        for n in range(n_initial_models):
            print(f"Epoch = 1, Initial model = {n}")
            # reset model
            model = FullNet(n_channel,n_constituent,dropout_p,device).to(device=device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            loss_weights = get_loss_weights(1)

            train_loop(train_dataloader, model, optimizer, nl.openbox_rmse, 
                       loss_weights, device)

            loss = test_loop(validation_dataloader, model, loss_functions, 
                             loss_names, loss_weights, device)

            if loss[0] < best_loss:
                best_loss = loss[0]
                n_best = n
                torch.save(
                    {
                    'epoch': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, 
                    model_filename + 'i' + str(n).zfill(2))
                print(f' Wrote Initial Model: {n}')
    
    else:
        if n_start == 1:
            model_filename_input = f'{model_filename}i' + str(n_best).zfill(2)
        else:
            model_filename_input = model_filename + str(n_start).zfill(3)
        checkpoint = torch.load(model_filename_input)
        print(f"Loaded Model: {model_filename_input}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        last_loss_weights = []
        n_best = n_start

        for n in range(n_start, n_epochs):
            print(f"Epoch = {n}")

            loss_weights = get_loss_weights(n)

            if loss_weights != last_loss_weights:
                last_loss_weights = loss_weights
                print(f"New loss weights = {loss_weights}")
                # Reset best loss
                n_best = n
                best_loss = 1.0e+08
                n_elapsed_best = 0

                dropout_p = get_dropout(n, n_epochs)
                if dropout_p != last_dropout_p:
                    last_dropout_p = dropout_p
                    model.reset_dropout(dropout_p)
                    print(f"New dropout: {dropout_p}")

                train_loop(train_dataloader, model, optimizer, device)

                loss = test_loop(validation_dataloader, model, loss_functions, 
                                 loss_names, loss_weights, device)

                is_write_model = False
                if n % checkpoint_period == 0:
                    is_write_model = True

                if loss[0] < best_loss:
                    best_loss = loss[0]
                    if n - n_best > n_elapsed_best:
                        n_elapsed_best = n - n_best
                    print(f"Epoch {n}, Improved Loss = {loss[0]}")
                    print(F"Elapsed epochs: current = {n - n_best} max ={n_elapsed_best}")
                    n_best = n
                    if n > n_windup:
                        is_write_model = True

                if is_write_model:
                    torch.save({
                        'epoch': n,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, 
                        model_filename + str(n).zfill(3))
                    print(f' Wrote Model: epoch = {n}')

                if n - n_best >= n_stop_training and n > n_windup:
                    n_elapsed_best = n - n_best
                    print(f"Max elapsed = {n_elapsed_best}")
                    print(f"Reached n_stop_training = {n_stop_training}")
                    torch.save({
                        'epoch': n,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, 
                        model_filename + str(n).zfill(3))
                    print(f' Wrote Model: epoch = {n}')
                    break

            print("Done!")

if __name__ == "__main__":
    train_network()