from netCDF4 import Dataset
import numpy as np
import time
from typing import List
import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn.functional as F

from RT_data_hws import absorbed_flux_to_heating_rate
import RT_sw_data

# Used only for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
t_direct_scattering = 0.0
t_direct_split = 0.0
t_scattering_v2_tau = 0.0
t_extinction = 0.0
t_total = 0.0
t_train = 0.0
t_loss = 0.0
t_grad = 0.0
t_backward = 0.0

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

    def __init__(self, n_input, n_hidden: List[int], n_output, dropout_p, device, 
                 lower=-0.1, upper=0.1, bias=True):
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
        
        self.input_weight = nn.parameter.Parameter(
            weight_values, requires_grad=True)

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
                weights_values, requires_grad=True)

        template = torch.ones(
            (4,3), device=device, dtype=torch.float32)
        self.output_filter = torch.block_diag (
            template2,template2,template2,template2, 
            template2,template2,template2,template2)
            
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

        # Repeat for clear case

        #tau_clear = self.clear_sky_mask * tau
        # f = number of features
        # [i,channels,f]

        #e_split_clear = self.diffuse_scattering(tau_clear)
        #n = e_split_clear.shape[0]
        
        # [i,channels,3, m]


        #e_split_clear = torch.reshape(e_split_clear,
        #                               (n, self.n_channel,
        #                                self.n_scattering_nets, 3))
        # [i,channels, m, 3]
        #e_split_clear = F.softmax(e_split_clear,dim=-1)

        #e_split_clear = self.diffuse_selection(e_split_clear)

        # [i, channels, 1, 3]

        #e_split_clear = torch.squeeze(e_split_clear, dim=-2)

        # [i,channels,3]  

        #e_split_clear = F.softmax(e_split_clear, dim=-1)

        layers = [t_full_direct, t_full_diffuse, e_split_full_direct, 
                  e_split_full_diffuse] #tau_full_total, t_clear, e_split_clear, tau_clear_total]

        return layers


class MultiReflection(nn.Module):
    """ 
    Computes each layer's radiative coefficients by accounting
    for interaction (multireflection) with all other layers using the 
    Adding-Doubling method (no learning).
    """

    def __init__(self):
        super(MultiReflection, self).__init__()

    def _adding_doubling (
        self, 
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

            See class Propagation below for how the multi-reflection
            coefficients are used to propagate radiation 
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
    We only need to propagate flux in a single pass
    since the radiative properties account for
    multi reflection

    Consider two downward fluxes entering the layer: 
                flux_direct, flux_diffuse

    Downward Direct Flux Transmitted = flux_direct * t_direct
    Downward Diffuse Flux Transmitted = 
                    flux_direct * t_multi_direct + 
                    flux_diffuse * (t_diffuse + t_multi_diffuse)

    Upward Flux from Top Layer = flux_direct * r_layer_multi_direct +
                            flux_diffuse * r_layer_multi_diffuse

    Upward Flux into Top Layer = 
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
        torch.cuda.synchronize()
        t_0 = time.time()
        #print(f"x_layers.shape = {x_layers.shape}")
        #9 constituents: lwc, ciw, h2o, o3, co2,  o2, n2o, ch4, co,  -no2?, 
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
        
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_total
        t_total += t_1 - t_0
        return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]


class FullNetInternals(nn.Module):
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

    def __init__(self, n_channel, n_constituent, dropout_p, device):
        super(FullNetInternals, self).__init__()
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
        torch.cuda.synchronize()
        t_0 = time.time()
        #print(f"x_layers.shape = {x_layers.shape}")
        #9 constituents: lwc, iwc, h2o, o3, co2,  o2, n2o, ch4, co,  -no2?, 
        (temperature_pressure, 
        constituents) = (x_layers[:,:,0:2], 
                        x_layers[:,:,2:10])
        
        mu_direct = x_surface[:,0] 

        one = torch.ones((1,),dtype=torch.float32,
                        device=self.device)
        
        mu_diffuse_original = torch.sigmoid(self.mu_diffuse_net(one))
        mu_diffuse = mu_diffuse_original.reshape((-1,1,1))
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

        # extinguished layers[i,layers,channels,3]
        t_direct, t_diffuse, e_split_direct,e_split_diffuse = layers
        # scattering fraction per channel
        s_direct_channels = (1.0 - t_direct) * (e_split_direct[:,:,:,0] + e_split_direct[:,:,:,1])
        s_diffuse_channels = (1.0 - t_diffuse) * (e_split_diffuse[:,:,:,0] + e_split_diffuse[:,:,:,1])
        #

        (multireflected_layers, 
        upward_reflection_toa) = self.multireflection_net((layers,x_surface,))

        channel_split = F.softmax(self.spectral_net(one),dim=-1) 

        r_toa = upward_reflection_toa * channel_split.reshape((1,-1))
        # sum over channels
        r_toa = torch.sum(r_toa,dim=1,keepdim=False)
        #

        flux_direct = torch.unsqueeze(channel_split,dim=0) * mu_direct.reshape((-1,1)) * self.solar_constant

        flux_diffuse = torch.zeros((mu_direct.shape[0], self.n_channel),
                                        dtype=torch.float32,
                                        device=self.device)
        input_flux = [flux_direct, flux_diffuse]

        flux = self.propagation_net((multireflected_layers, 
                                    upward_reflection_toa,
                                    input_flux))

        channel_split = channel_split.reshape((1,1,-1))
        # Weight the channels appropriately
        # (i, 1, n_channels)
        s_direct_channels = s_direct_channels * channel_split
        s_diffuse_channels = s_diffuse_channels * channel_split

        t_direct_total = t_direct * channel_split
        t_diffuse_total = t_diffuse * channel_split
        # Scattering fraction for entire layer
        s_direct = torch.sum(s_direct_channels,dim=2,keepdim=False)
        s_diffuse = torch.sum(s_diffuse_channels,dim=2,keepdim=False)
        t_direct_total = torch.sum(t_direct_total,dim=2,keepdim=False)
        t_diffuse_total = torch.sum(t_diffuse_total,dim=2,keepdim=False)

        (flux_down_direct, flux_down_diffuse, flux_up_diffuse, 
        flux_absorbed) = flux
        
        torch.cuda.synchronize()
        t_1 = time.time()
        global t_total
        t_total += t_1 - t_0

        internal_data = [x_layers[:,:,2], x_layers[:,:,3], x_layers[:,:,5], mu_diffuse_original, s_direct, s_diffuse, r_toa, x_surface[:,1],
                         mu_direct, t_direct_total, t_diffuse_total, x_layers[:,:,4]]
        predicted_data = [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]

        return predicted_data, internal_data
        #return [flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_absorbed]

def loss_heating_rate_2(flux_down_true, flux_down_pred,
                           delta_pressure):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:])

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:])
    
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    loss = torch.sqrt(torch.mean(torch.square(heat_true - heat_pred),
                                  dim=(0,1),keepdim=False))
    return loss

def layered_rmse(reference, pred):
    loss = torch.sqrt(torch.mean(torch.square(reference - pred),
                                  dim=(0,),keepdim=False))
    return loss

def layered_bias(reference, pred):
    loss = torch.mean(reference - pred,dim=(0,),keepdim=False)
    return loss

def layered_mae(reference, pred):
    loss = torch.mean(torch.abs(reference - pred),
                                  dim=(0,),keepdim=False)
    return loss

def geographic_rmse(reference, pred, sites, number_of_sites):
    # mean over layers
    loss = torch.mean(torch.square(reference - pred),
                                  dim=(1,),keepdim=False)
    sum = torch.zeros((number_of_sites,),dtype=torch.float32)
    count = torch.zeros((number_of_sites,),dtype=torch.int32)

    for i, site in enumerate(sites):
        count[site] = count[site] + 1
        sum[site] = sum[site] + loss[i]

    return sum, count

def geographic_bias(reference, pred, sites, number_of_sites):
    loss = torch.mean(reference - pred,dim=(1,),keepdim=False)

    sum = torch.zeros((number_of_sites,),dtype=torch.float32)
    count = torch.zeros((number_of_sites,),dtype=torch.int32)

    for i, site in enumerate(sites):
        count[site] = count[site] + 1
        sum[site] = sum[site] + loss[i]

    return sum, count


def loss_layered_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                            delta_pressure, loss_metric_function):
        
    flux_absorbed_true = (flux_down_true[:,:-1] -
                            flux_down_true[:,1:] + 
                            flux_up_true[:,1:] -
                            flux_up_true[:,:-1])
                            

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                            flux_down_pred[:,1:] + 
                            flux_up_pred[:,1:] -
                            flux_up_pred[:,:-1])
    
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                            delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                            delta_pressure)
    loss = loss_metric_function(heat_true, heat_pred)
    #loss = torch.sqrt(torch.mean(torch.square(heat_true - heat_pred),
    #                            dim=(0,),keepdim=False))
    return loss

def loss_geographic_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                            delta_pressure, sites, number_of_sites, loss_metric_function):
        
    flux_absorbed_true = (flux_down_true[:,:-1] -
                            flux_down_true[:,1:] + 
                            flux_up_true[:,1:] -
                            flux_up_true[:,:-1])
                            

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                            flux_down_pred[:,1:] + 
                            flux_up_pred[:,1:] -
                            flux_up_pred[:,:-1])
    
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                            delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                            delta_pressure)

    loss, count = loss_metric_function(heat_true, heat_pred, sites, number_of_sites)
    #loss = torch.sqrt(torch.mean(torch.square(heat_true - heat_pred),
    #                            dim=(0,),keepdim=False))
    return loss, count

def loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                           delta_pressure):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:] + 
                             flux_up_true[:,1:] -
                             flux_up_true[:,:-1])

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:] + 
                             flux_up_pred[:,1:] -
                             flux_up_pred[:,:-1])
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    loss = torch.sqrt(torch.mean(torch.square(heat_true - heat_pred),
                                  dim=(0,1),keepdim=False))
    return loss


def individual_squared_loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                           delta_pressure):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:] + 
                             flux_up_true[:,1:] -
                             flux_up_true[:,:-1])

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:] + 
                             flux_up_pred[:,1:] -
                             flux_up_pred[:,:-1])
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    loss = torch.square(heat_true - heat_pred)
    return loss

def bias_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                           delta_pressure):
    
    flux_absorbed_true = (flux_down_true[:,:-1] -
                             flux_down_true[:,1:] + 
                             flux_up_true[:,1:] -
                             flux_up_true[:,:-1])

    flux_absorbed_pred = (flux_down_pred[:,:-1] -
                             flux_down_pred[:,1:] + 
                             flux_up_pred[:,1:] -
                             flux_up_pred[:,:-1])
    heat_true = absorbed_flux_to_heating_rate(flux_absorbed_true, 
                                              delta_pressure)
    heat_pred = absorbed_flux_to_heating_rate(flux_absorbed_pred, 
                                              delta_pressure)
    bias = torch.mean(heat_true - heat_pred,
                                  dim=(0,1),keepdim=False)
    return bias

def loss_direct_heating_rate_wrapper(data, y_pred, loss_weights):
    _, _, delta_pressure, y_true, _, = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]

    hr_loss = loss_heating_rate_2(flux_down_direct_true, flux_down_direct_pred, delta_pressure)
    
    return hr_loss

def loss_diffuse_heating_rate_wrapper(data, y_pred, loss_weights):
    _, _, delta_pressure, y_true, _, = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(_, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true

    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    hr_loss = loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_loss

def loss_full_heating_rate_wrapper(data, y_pred, loss_weights):
    _, _, delta_pressure, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    hr_loss = loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_loss

def individual_squared_loss_heating_rate_wrapper(data, y_pred):
    _, _, delta_pressure, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    hr_loss = individual_squared_loss_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_loss

def loss_geographic_heating_rate_maker(loss_metric_function, number_of_sites):
    def loss_geographic_heating_rate_wrapper(data, y_pred, loss_weights):
        _, _, delta_pressure, y_true, sites = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        hr_loss, hr_count = loss_geographic_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure, sites, number_of_sites, loss_metric_function)
        
        return hr_loss, hr_count
    return loss_geographic_heating_rate_wrapper

def loss_layered_heating_rate_maker(loss_metric_function):
    def loss_layered_heating_rate_wrapper(data, y_pred, loss_weights):
        _, _, delta_pressure, y_true, _ = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        hr_loss = loss_layered_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure, loss_metric_function)
        
        return hr_loss
    return loss_layered_heating_rate_wrapper

def loss_layered_flux_maker(loss_metric_function, is_down):
    def loss_layered_flux_wrapper(data, y_pred, loss_weights):
        _, _, delta_pressure, y_true, _ = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        if is_down:
            flux_error = loss_metric_function(flux_down_true, flux_down_pred)
        else:
            flux_error = loss_metric_function(flux_up_true, flux_up_pred)
        
        return flux_error
    return loss_layered_flux_wrapper

def loss_geographic_flux_maker(loss_metric_function, number_of_sites, is_down):
    def loss_geographic_flux_wrapper(data, y_pred, loss_weights):
        _, _, _, y_true, sites = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        if is_down:
            flux_error, flux_count = loss_metric_function(flux_down_true, flux_down_pred, sites, number_of_sites)
        else:
            flux_error, flux_count = loss_metric_function(flux_up_true, flux_up_pred, sites, number_of_sites)
        
        return flux_error, flux_count
    return loss_geographic_flux_wrapper

def bias_full_heating_rate_wrapper(data, y_pred, loss_weights):
    _, _, delta_pressure, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    hr_bias = bias_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure)
    
    return hr_bias

def loss_flux_2(flux_true, flux_pred):  

    flux_loss = torch.sqrt(torch.mean(torch.square(flux_pred - flux_true), 
                       dim=(0,1), keepdim=False))

    return flux_loss

def loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred):  

    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_loss = torch.sqrt(torch.mean(torch.square(flux_pred - flux_true), 
                       dim=(0,1), keepdim=False))

    return flux_loss


def bias_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred):  

    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    flux_bias = torch.mean(flux_pred - flux_true, 
                       dim=(0,1), keepdim=False)

    return flux_bias

def bias_flux_2(flux_true, flux_pred):  

    flux_bias = torch.mean(flux_pred - flux_true, 
                       dim=(0,1), keepdim=False)

    return flux_bias


def loss_direct_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, _, _, _) = y_pred
    flux_down_direct_true = y_true[:,:,0]
    #(flux_down_direct_true, _, _, _, _, _) = y_true

    loss = loss_flux_2(flux_down_direct_true, flux_down_direct_pred)
    return loss

def loss_diffuse_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (_, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(_, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    loss = loss_flux(flux_down_diffuse_true, flux_up_diffuse_true,flux_down_diffuse_pred, flux_up_diffuse_pred)
    return loss

def loss_full_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    loss = loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    return loss

def mu_selector_maker_loss_full_flux(mu_threshold):
    def loss_function(data, y_pred, loss_weights):
        _, x_surface, _, y_true, _ = data
        mu = x_surface[:,0]
        selection = mu < mu_threshold
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        flux_down_direct_pred = flux_down_direct_pred[selection]
        flux_down_diffuse_pred = flux_down_diffuse_pred[selection]
        flux_up_diffuse_pred = flux_up_diffuse_pred[selection]
        #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_direct_true = y_true[selection,:,0]
        flux_down_diffuse_true = y_true[selection,:,1]
        flux_up_diffuse_true = y_true[selection,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        loss = loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
        return loss
    return loss_function

def loss_down_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    loss = loss_flux_2(flux_down_true, flux_down_pred)
    return loss

def loss_up_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    loss = loss_flux_2(flux_up_true, flux_up_pred)
    return loss

def bias_full_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    bias = bias_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred)
    return bias

def bias_down_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    bias = bias_flux_2(flux_down_true, flux_down_pred)
    return bias

def bias_up_flux_wrapper(data, y_pred, loss_weights):
    _, _, _, y_true, _ = data
    (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
    #(flux_down_direct_true, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
    flux_down_direct_true = y_true[:,:,0]
    flux_down_diffuse_true = y_true[:,:,1]
    flux_up_diffuse_true = y_true[:,:,2]

    flux_down_true = flux_down_direct_true + flux_down_diffuse_true
    flux_up_true = flux_up_diffuse_true

    flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
    flux_up_pred = flux_up_diffuse_pred

    bias = bias_flux_2(flux_up_true, flux_up_pred)
    return bias

def loss_henry_wrapper(data, y_pred, loss_weights):

    loss_direct_flux = loss_direct_flux_wrapper(data, y_pred, loss_weights)    
    loss_diffuse_flux = loss_diffuse_flux_wrapper(data, y_pred, loss_weights)

    loss_direct_heating_rate = loss_direct_heating_rate_wrapper(data, y_pred, loss_weights)
    loss_diffuse_heating_rate = loss_diffuse_heating_rate_wrapper(data, y_pred, loss_weights)

    # 0-200
    #w1 = 2.0
    #w2 = 1.0
    #w3 = 0.5
    #w4 = 0.25

    # 200 +
    #w1 = 1.0
    #w2 = 1.0
    #w3 = 0.5
    #w4 = 0.5

    # 285 +
    #w1 = 1.0
    #w2 = 1.0
    #w3 = 1.0
    #w4 = 1.0

    # 360 +
    #w1 = 1.0
    #w2 = 1.0
    #w3 = 2.0
    #w4 = 2.0

    # 515 +
    #w1 = 1.0
    #w2 = 1.0
    #w3 = 0.5
    #w4 = 0.5

    w1 = loss_weights[0]
    w2 = loss_weights[1]
    w3 = loss_weights[2]
    w4 = loss_weights[3]

    loss = (1.0 / (w1 + w2 + w3 + w4)) * (w1 * loss_direct_flux + w2 * loss_diffuse_flux + w3 * loss_direct_heating_rate + w4 * loss_diffuse_heating_rate)

    return loss


def loss_henry_wrapper_2(data, y_pred, loss_weights):
    # Just flux and heating rate
    loss_flux = loss_full_flux_wrapper(data, y_pred, loss_weights)    

    loss_heating_rate = loss_full_heating_rate_wrapper(data, y_pred, loss_weights)

    w1 = loss_weights[0]
    w2 = loss_weights[1]


    loss = (1.0 / (w1 + w2)) * (w1 * loss_flux + w2 * loss_heating_rate)

    return loss



def train_loop(dataloader, model, optimizer, loss_function, loss_weights, device):
    """ Generic training loop """

    torch.cuda.synchronize()
    t_1 = time.time()
    model.train()

    loss_string = "Training Loss: "
    for batch, data in enumerate(dataloader):
        data = [x.to(device) for x in data]
        y_pred = model(data)
        torch.cuda.synchronize()
        t_0 = time.time()
        loss = loss_function(data, y_pred, loss_weights)
        torch.cuda.synchronize()
        t_01 = time.time()
        global t_loss
        t_loss += t_01 - t_0

        if False:
            with torch.autograd.profiler.profile(use_cuda=True,
                                             use_cpu=True,
                                             with_modules=True,
                                             with_stack=True) as prof:
                loss.backward()
            print(prof.key_averages(group_by_stack_n=8).table(sort_by="cuda_time_total"))
        else:
            loss.backward()
        torch.cuda.synchronize()
        t_03 = time.time()
        global t_backward
        t_backward += t_03 - t_01
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            loss_value = loss.item()
            loss_string += f" {loss_value:.9f}"
        torch.cuda.synchronize()
        t_02 = time.time()
        global t_grad
        t_grad += t_02 - t_01
    #print (loss_string)
    torch.cuda.synchronize()
    t_2 = time.time()
    global t_train
    t_train += t_2 - t_1

def test_loop(dataloader, model, loss_functions, loss_names, loss_weights, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)
    elapsed_time_ms = 0.0

    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            start_event.record()
            y_pred = model(data)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, loss_weights).item()

    loss /= num_batches

    print(f"Elapsed time: {elapsed_time_ms:.2f} ms")

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
    print("")

    return loss

# computes an error metric for each layer
def test_layers_loop(dataloader, model, loss_functions, loss_names, loss_weights, is_flux, device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    # Determining number of layers
    dataset = dataloader.dataset
    sample, _, _, _, _ = dataset[0]
    sample_shape = sample.shape

    # Loss for each atmospheric column
    if is_flux:
        loss = np.zeros((len(loss_functions), sample_shape[0] + 1), dtype=np.float32)
    else:
        loss = np.zeros((len(loss_functions), sample_shape[0]), dtype=np.float32)
    with torch.no_grad():
        for data in dataloader:
            data = [x.to(device) for x in data]
            y_pred = model(data)
            for i, loss_fn in enumerate(loss_functions):
                loss[i,:] += loss_fn(data, y_pred, loss_weights).numpy()

    loss /= num_batches

    print(f"Test Error: ")
    for i, values in enumerate(loss):
        print(f" {loss_names[i]}:")
        for j, value in enumerate(values):
            print (f"   {j}. {value:.8f}")
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
                tmp_loss, tmp_count = loss_fn(data, y_pred,loss_weights)
                loss[i,:] += tmp_loss.numpy()
                count[i,:] += tmp_count.numpy()

    for i, name in enumerate(loss_names):
        loss[i,:] = loss[i,:] / np.float32(count[i,:])
        if name.find("rmse") > 0:
            print(f"Computing RMSE for {name}")
            loss[i,:] = np.sqrt(loss[i,:])
        else:
            print(f"Computing bias for {name}")

    dt = Dataset(loss_file_name, "w")
    dim1 = dt.createDimension("sites",number_of_sites)
    for i, name in enumerate(loss_names):
        var = dt.createVariable(name,"f4",("sites",))
        var[:] = loss[i,:]

    dt.close()
    return loss

def test_loop_internals (dataloader, model, loss_functions, loss_names, loss_weights,device):
    """ Generic testing / evaluation loop """
    model.eval()
    num_batches = len(dataloader)

    loss = np.zeros(len(loss_functions), dtype=np.float32)

    lwp = []
    iwp = []
    o3 = []
    mu_diffuse = []
    s_direct   = []
    s_diffuse   = []
    r_toa = []
    r_surface = []
    mu_direct = []
    t_direct = []
    t_diffuse = []
    h2o = []
    #squared_loss = []

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
            #squared_loss.append(individual_squared_loss_heating_rate_wrapper(data,y_pred))
            for i, loss_fn in enumerate(loss_functions):
                loss[i] += loss_fn(data, y_pred, loss_weights).item()

    loss /= num_batches

    print(f"Test Error: ")
    for i, value in enumerate(loss):
        print(f" {loss_names[i]}: {value:.8f}")
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

    internal_data = [lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse, r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o]

    return loss, internal_data

def get_loss_weights(t):
    if t <= 200:
        loss_weights = [2.0, 1.0, 0.5, 0.25]
    elif t <= 285:
        loss_weights = [1.0, 1.0, 0.5, 0.5]
    elif t <= 360:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    elif t <= 515:
        loss_weights = [1.0, 1.0, 1.0, 1.0]
    elif t <= 600:
        loss_weights = [1.0, 1.0, 0.5, 0.5]
    else:
        loss_weights = [0.5, 2.0, 1.0, 1.0]
    return loss_weights

def get_dropout (t):
    dropout_schedule = (0.0, 0.07, 0.1, 0.15, 0.2, 0.15, 0.1, 0.07, 0.0, 0.0) 
    dropout_epochs =   (-1, 40, 60, 70,  80, 90,  105, 120, 135, n_max_epochs + 1)

    dropout_index = next(i for i, x in enumerate(dropout_epochs) if t <= x) - 1
    dropout_p = dropout_schedule[dropout_index]
    return dropout_p

def train_full_dataloader():

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

    datadir     = "/data-T1/hws/tmp/"
    train_input_dir = f"/data-T1/hws/CAMS/processed_data/train/2008/"
    cross_input_dir = "/data-T1/hws/CAMS/processed_data/cross_validation/2008/"
    months = [str(m).zfill(2) for m in range(1,13)]

    # These cannot be changed. Assumed by OpticalDepth
    n_channel = 42
    n_constituent = 8
    ############################################

    t_start = 0
    version_name = "v1.v1."  

    #### Meta parameters #############
    batch_size = 1024
    max_elapsed_epochs = 50
    windup_best_loss = 200
    n_initial_models = 4
    n_max_epochs = 2000
    checkpoint_period = 5
    ##################################

    best_loss_index = -1
    best_loss = 1.0e+08

    elapsed_epochs = 0

    train_input_files = [f'{train_input_dir}nn_input_sw-train-2008-{month}.nc' for month in months]
    cross_input_files = [f'{cross_input_dir}nn_input_sw-cross_validation-2008-{month}.nc' for month in months]

    filename_full_model = datadir + f"/Torch.SW.{version_name}" 

    train_dataset = RT_sw_data.RTDataSet(train_input_files, is_clear_sky=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size, shuffle=False, num_workers=1)
    
    validation_dataset = RT_sw_data.RTDataSet(cross_input_files, is_clear_sky=False)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size, shuffle=False, num_workers=1)

    loss_functions = (
        loss_henry_wrapper,  loss_full_flux_wrapper, loss_direct_flux_wrapper, loss_diffuse_flux_wrapper, bias_full_flux_wrapper, 
        loss_full_heating_rate_wrapper, loss_direct_heating_rate_wrapper, loss_diffuse_heating_rate_wrapper)

    loss_names = (
        "Loss", "Full Flux Loss", "Direct Flux Loss",
        "Diffuse Flux Loss", "Flux Bias", 
        "Full Heating Rate Loss","Direct Heating Rate Loss", 
        "Diffuse Heating Rate Loss")

    if t_start == 0:
        # Start from the beginning

        for n in range(n_initial_models):
            # reset model
            model = FullNet(n_channel,n_constituent,dropout_p,device).to(device=device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            loss_weights = get_loss_weights(1)

            train_loop(train_dataloader, model, optimizer, 
                            loss_henry_wrapper, loss_weights,
                            device)

            loss = test_loop(validation_dataloader, model, loss_functions, loss_names, loss_weights, device)

            if loss[0] < best_loss:
                best_loss = loss[0]
                best_loss_index = n
                torch.save(
                    {
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, 
                    filename_full_model + 'i' + str(n).zfill(2))
                print(f' Wrote Initial Model: {n}')
    
    else:
        if t_start == 1:
            filename_full_model_input = f'{filename_full_model}i' + str(best_loss_index).zfill(2)
        else:
            filename_full_model_input = filename_full_model + str(t_start).zfill(3)
        checkpoint = torch.load(filename_full_model_input)
        print(f"Loaded Model: epoch = {filename_full_model_input}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        last_loss_weights = []
        best_loss_index = t_start
        last_dropout_p = 0.0

        for t in range(t_start, n_max_epochs):
            loss_weights = get_loss_weights(t)

            if loss_weights != last_loss_weights:
                last_loss_weights = loss_weights
                print(f"New loss weights = {loss_weights}")
                # Reset best loss
                best_loss_index = t
                best_loss = 1.0e+08

                dropout_p = get_dropout(t)
                if dropout_p != last_dropout_p:
                    last_dropout_p = dropout_p
                    model.reset_dropout(dropout_p)
                    print(f"New dropout: {dropout_p}")

                train_loop(train_dataloader, model, optimizer, 
                        device)

                loss = test_loop(validation_dataloader, model, loss_functions, loss_names, loss_weights, device)

                if t % checkpoint_period == 0 or (loss[0] < best_loss and t > windup_best_loss) or (t - best_loss_index > max_elapsed_epochs and  t > windup_best_loss):

                    torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename_full_model + str(t).zfill(3))
                    print(f' Wrote Model: epoch = {t}')

                if loss[0] < best_loss:
                    best_loss = loss[0]
                    if t - best_loss_index > elapsed_epochs:
                        elapsed_epochs = t - best_loss_index
                    print(f"Epoch {t}, Improved Loss = {loss[0]}")
                    print(F"Elapsed epochs: current = {t - best_loss_index} max ={elapsed_epochs}")
                    best_loss_index = t


                if t - best_loss_index >= max_elapsed_epochs and t > windup_best_loss:
                    current_max_epochs = t - best_loss_index
                    print(f"Max elapsed = {current_max_epochs}")
                    print(f"Reached max_elapsed_epochs = {max_elapsed_epochs}")
                    break

            print("Done!")



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_internal_data(internal_data, output_file_name):
    import xarray as xr
    lwp, iwp, o3, mu_diffuse, s_direct, s_diffuse, r_toa, r_surface, mu_direct, t_direct, t_diffuse, h2o = internal_data

    shape = lwp.shape
    shape2 = lwp.numpy().shape

    example = np.arange(shape[0])
    layer = np.arange(shape[1])


    #lwp = xr.DataArray(lwp, coords=[time,site,layer], dims=("time","site","layer"), name="lwp")

    #iwp = xr.DataArray(iwp, coords=[time,site,layer], dims=("time","site","layer"), name="iwp")

    #r = xr.DataArray(r, coords=[time,site,layer],dims=("time","site","layer"), name="r")

    mu_diffuse = mu_diffuse.numpy().flatten()
    
    mu_diffuse_n = np.arange(mu_diffuse.shape[0])
    mu_direct = mu_direct.numpy()
    #s1 = np.shape(mu_direct)
    #mu_direct = np.reshape(mu_direct, (s1[0], s1[1]*s1[2]))

    rs_direct = s_direct.numpy()
    rs_diffuse = s_diffuse.numpy()
    rr_toa = r_toa.numpy()
    rr_surface = r_surface.numpy()

    is_bad = np.isnan(rs_direct).any() or np.isnan(rs_diffuse).any()
    print(f"is bad = {is_bad}")

    ds = xr.Dataset(
        data_vars = {
            "lwp": (["example","layer"], lwp.numpy()),
            "iwp": (["example","layer"], iwp.numpy()),
            "o3": (["example","layer"], o3.numpy()),
            "mu_diffuse"  : (["mu_diffuse_n"], mu_diffuse),
            "mu_direct"  : (["example"], mu_direct),
            "s_direct": (["example","layer"], rs_direct),
            "s_diffuse": (["example","layer"], rs_diffuse),
            "r_toa" : (["example"], rr_toa),
            "r_surface" : (["example"], rr_surface),
            "t_direct": (["example","layer"], t_direct.numpy()),
            "t_diffuse": (["example","layer"], t_diffuse.numpy()),
            "h2o": (["example","layer"], h2o.numpy()),
            },
         coords = {
             "example" : example,
             "layer" : layer,
             "mu_diffuse_n" : mu_diffuse_n,
         },
    )

    ds.to_netcdf(output_file_name)
    ds.close()


def test_full_dataloader():

    if False:
        print("Pytorch version:", torch.__version__)
        device = "cpu"
        print(f"Using {device} device")

    else:
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

    datadir     = "/data-T1/hws/tmp/"
    batch_size = 1024
    n_channel = 42
    n_constituent = 8

    is_use_internals = False

    is_mcica = False #True

    is_geographic_loss = False
    number_of_sites = 5120 #THIS SHOULD NOT BE HARDCODED!!!

    is_layered_loss = False
    is_flux = True  # only matters when is_layered_loss = True or is_geographic_loss
    is_down = False # only matters when is_layered_loss = True or is_geographic_loss

    is_clear_sky = False

    if is_mcica:
        version_name = "v1.v2."
    else:
        version_name = "v1.v1."  # expanded error
        #version_name = "v1.v4." # standard error


    if is_use_internals:
        model = FullNetInternals(n_channel,n_constituent,dropout_p=0,device=device)
    else:
        model = FullNet(n_channel,n_constituent,dropout_p=0,device=device)

    model = model.to(device=device)

    filename_full_model = datadir + f"/Torch.SW.{version_name}" # 

    loss_file_name =  datadir + f"/Loss_Torch.SW.{version_name}"
    if True:
        mode = "testing"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/testing/"
        years = ("2009", "2015", "2020")
    else:
        mode = "training"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/training/"
        years = ("2008", )

    for year in years:
        
        test_input_dir = f"{processed_data_dir}{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        if is_mcica:
            test_input_files = [f'{test_input_dir}nn_input_sw_mcica-{mode}-{year}-{month}.nc' for month in months]
        else:
            test_input_files = [f'{test_input_dir}nn_input_sw-{mode}-{year}-{month}.nc' for month in months]
        #test_input_files = ["/data-T1/hws/tmp/RADSCHEME_data_g224_CAMS_2015_true_solar_angles.2.nc"]


        test_dataset = RT_sw_data.RTDataSet(test_input_files, is_clear_sky=is_clear_sky)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)

        if not is_flux:
            loss_layered_heating_rate_rmse = loss_layered_heating_rate_maker(layered_rmse)
            loss_layered_heating_rate_bias = loss_layered_heating_rate_maker(layered_bias)
            loss_layered_heating_rate_mae = loss_layered_heating_rate_maker(layered_mae)
            layered_loss_functions = (loss_layered_heating_rate_rmse,loss_layered_heating_rate_bias, loss_layered_heating_rate_mae)


        else:
            loss_layered_flux_rmse = loss_layered_flux_maker(layered_rmse, is_down)
            loss_layered_flux_bias = loss_layered_flux_maker(layered_bias, is_down)
            loss_layered_flux_mae = loss_layered_flux_maker(layered_mae, is_down)
            layered_loss_functions = (loss_layered_flux_rmse,loss_layered_flux_bias, loss_layered_flux_mae)


        if is_flux:
            if is_down:
                layered_loss_names = ("downwelling flux rmse","downwelling flux bias", "downwelling flux mae")
            else:
                layered_loss_names = ("upwelling flux rmse","upwelling flux bias", "upwelling flux mae")
        else:
                layered_loss_names = ("heating rate rmse","heating rate bias", "heating rate mae")

        loss_geographic_heating_rate_rmse = loss_geographic_heating_rate_maker(geographic_rmse, number_of_sites)
        loss_geographic_heating_rate_bias = loss_geographic_heating_rate_maker(geographic_bias, number_of_sites)

        loss_geographic_down_flux_rmse = loss_geographic_flux_maker(geographic_rmse, number_of_sites, is_down=True)
        loss_geographic_down_flux_bias = loss_geographic_flux_maker(geographic_bias, number_of_sites, is_down=True)

        loss_geographic_up_flux_rmse = loss_geographic_flux_maker(geographic_rmse, number_of_sites, is_down=False)
        loss_geographic_up_flux_bias = loss_geographic_flux_maker(geographic_bias, number_of_sites, is_down=False)

        geographic_loss_functions = (loss_geographic_heating_rate_rmse,loss_geographic_heating_rate_bias,
        loss_geographic_down_flux_rmse,loss_geographic_down_flux_bias,
        loss_geographic_up_flux_rmse,loss_geographic_up_flux_bias)

        geographic_loss_names = ("heating_rate_rmse","heating_rate_bias", "downwelling_flux_rmse","downwelling_flux_bias","upwelling_flux_rmse","upwelling_flux_bias")

        loss_flux_0_0025 = mu_selector_maker_loss_full_flux(0.0025)
        loss_flux_0_01 = mu_selector_maker_loss_full_flux(0.01)
        loss_flux_0_05 = mu_selector_maker_loss_full_flux(0.05)
        loss_flux_0_10 = mu_selector_maker_loss_full_flux(0.10)

        loss_functions = (loss_henry_wrapper, #
                          #loss_henry_wrapper_2, 
                          loss_full_flux_wrapper, loss_direct_flux_wrapper, loss_diffuse_flux_wrapper, 
                          bias_full_flux_wrapper, loss_down_flux_wrapper, loss_up_flux_wrapper, bias_down_flux_wrapper, bias_up_flux_wrapper, 
                          loss_full_heating_rate_wrapper, loss_direct_heating_rate_wrapper, loss_diffuse_heating_rate_wrapper,
                          bias_full_heating_rate_wrapper,
                          loss_flux_0_0025,
                          loss_flux_0_01,
                          loss_flux_0_05,
                          loss_flux_0_10)
        loss_names = ("Loss", "Full Flux Loss", "Direct Flux Loss","Diffuse Flux Loss","Flux Bias", "Flux Down", "Flux up", "Flux Down Bias", "Flux up Bias ", "Full Heating Rate Loss","Direct Heating Rate Loss", "Diffuse Heating Rate Loss", "Heating Rate Bias","loss_flux_0_0025", "loss_flux_0_01",  "loss_flux_0_05", "loss_flux_0_10")

        print(f"Testing error, Year = {year}")

        for t in range(596, 601, 5): #.v1.v1 expanded loss
        # #range(618, 623, 5): #v1.v4 standard loss
        #for t in range(285, 650, 5): #range(618, 623, 5):
        #for t in range(140, 650, 5): #range(618, 623, 5):
        #for t in range(140, 285, 5): #range(618, 623, 5):
        #for t in range(618, 623, 5):

            if True:
                if t <= 200:
                    loss_weights = [2.0, 1.0, 0.5, 0.25]
                elif t <= 285:
                    #loss_weights = [1.0, 1.0]
                    loss_weights = [1.0, 1.0, 0.5, 0.5]
                elif t <= 360:
                    loss_weights = [1.0, 1.0, 1.0, 1.0]
                elif t <= 515:
                    loss_weights = [1.0, 1.0, 2.0, 2.0]
                else:
                    loss_weights = [1.0, 1.0, 0.5, 0.5]
            else:
                loss_weights = [1.0, 1.0]

            checkpoint = torch.load(filename_full_model + str(t).zfill(3), map_location=torch.device(device))
            print(f"Loaded Model: epoch = {t}")
            model.load_state_dict(checkpoint['model_state_dict'])

            #print(f"Total number of parameters = {n_parameters}", flush=True)
            #print(f"Spectral decomposition weights = {model.spectral_net.weight}", flush=True)

            num_batches = len(test_dataloader)

            if False:
                # Determining number of layers
                dataset = test_dataloader.dataset
                print(f"Number of Batches = {num_batches}")
                sample, _, _, _ = dataset[0]
                print(f'Dataset len = {len(dataset)}')
                sample_shape = sample.shape
                print(f'Sample Shape = {sample_shape}')

            if is_geographic_loss:
                loss = test_geographic_loop(test_dataloader, model, geographic_loss_functions, geographic_loss_names, loss_weights, number_of_sites, loss_file_name  + str(t).zfill(3) + f".{year}.nc", device)

                #write loss to file

            if is_layered_loss:
                loss = test_layers_loop(test_dataloader, model, layered_loss_functions, layered_loss_names, loss_weights, is_flux, device)

            if is_use_internals:
                loss, internal_data = test_loop_internals (test_dataloader, model, loss_functions, loss_names, loss_weights, device)
                write_internal_data(internal_data, output_file_name=test_input_dir + f"internal_output.sc_{version_name}_{t}.{year}.nc")
            elif not is_layered_loss and not is_geographic_loss:
                loss = test_loop (test_dataloader, model, loss_functions, loss_names, loss_weights, device)
    

if __name__ == "__main__":
    #train_direct_only()
    #train_full()
    #
    #test_full()

    #global t_direct_scattering = 0.0
    #global t_direct_split = 0.0
    #t_scattering_v2_tau = 0.0
    
    #train_full_dataloader()
    test_full_dataloader()

   