import sys
import numpy as np
import tensorflow as tf

from netCDF4 import Dataset
import xarray as xr
import random

def get_weight_profile(file_name):
        dt = xr.open_dataset(file_name)
        rsd = dt['rsd'].data
        s = rsd.shape
        rsd = np.reshape(rsd, (s[0]*s[1],s[2]))
        rsu = dt['rsu'].data
        rsu = np.reshape(rsu, (s[0]*s[1],s[2]))
        rsd0 = rsd[:,0]
        y = np.concatenate((np.expand_dims(rsd,2), np.expand_dims(rsu,2)),axis=2)
        y = y / np.reshape(rsd0, (-1, 1, 1))
        dt.close()
        return 1/y.mean(axis=0)

def get_col_dry(vmr_h2o, plev):
    grav = 9.80665
    m_dry = 0.028964
    m_h2o =  0.018016
    avogad = 6.02214076e23
    delta_plev = plev[:,1:] - plev[:,0:-1]
    # Get average mass of moist air per mole of moist air
    fact = 1.0 / (1. + vmr_h2o)
    m_air = (m_dry + m_h2o * vmr_h2o) * fact
    col_dry = 10.0 * np.float64(delta_plev) * avogad * np.float64(fact) / (1000.0 * m_air * 100.0 * grav)
    return np.float32(col_dry)



def load_radscheme_rnn(fname, predictand='rsu_rsd', scale_p_h2o_o3=True, \
                                return_p=False, return_coldry=False):
    # Load data for training a RADIATION SCHEME (RTE+RRTMGP) emulator,
    # where inputs are vertical PROFILES of atmospheric conditions (T,p, gas concentrations)
    # and outputs (predictand) are PROFILES of broadband fluxes (upwelling and downwelling)
    # argument scale_p_h2o_o3 determines whether specific gas optics inputs
    # (pressure, H2O and O3 )  are power-scaled similarly to Ukkonen 2020 paper
    # for a more normal distribution
    
    dat = Dataset(fname)
    
    if predictand not in ['rsu_rsd']:
        sys.exit("Supported predictands (second argument) : rsu_rsd..")
            
    # temperature, pressure, and gas concentrations...
    x_gas = dat.variables['rrtmgp_sw_input'][:].data  # (nexp,ncol,nlay,ngas+2)
    (nexp,ncol,nlay,nx) = x_gas.shape
    nlev = nlay+1
    ns = nexp*ncol # number of samples (profiles)
    if scale_p_h2o_o3:
        # Log-scale pressure, power-scale H2O and O3
        x_gas[:,:,:,1] = np.log(x_gas[:,:,:,1])
        vmr_h2o = x_gas[:,:,:,2].reshape(ns,nlay)
        x_gas[:,:,:,2] = x_gas[:,:,:,2]**(1.0/4) 
        x_gas[:,:,:,3] = x_gas[:,:,:,3]**(1.0/4)
    
    # plus surface albedo, which !!!FOR THIS DATA!!! is spectrally constant
    sfc_alb = dat.variables['sfc_alb'][:].data # (nexp,ncol)

    # plus by cosine of solar angle..
    mu0 = dat.variables['mu0'][:].data           # (nexp,ncol)
    # # ..multiplied by incoming flux
    # #  (ASSUMED CONSTANT)
    # toa_flux = dat.variables['toa_flux'][:].data # (nexp,ncol,ngpt)
    # ngpt = toa_flux.shape[-1]
    # for iexp in range(nexp):
    #     for icol in range(ncol):
    #         toa_flux[iexp,icol,:] = mu0[iexp,icol] * toa_flux[iexp,icol,:]
    
    lwp = dat.variables['cloud_lwp'][:].data
    iwp = dat.variables['cloud_iwp'][:].data

    # if predictand in ['broadband_rsu_rsd','broadband_rlu_rld']: 
    rsu = dat.variables['rsu'][:]
    rsd = dat.variables['rsd'][:]
        
    if np.size(rsu.shape) != 3:
        sys.exit("Invalid array shapes, RTE output should have 3 dimensions")
    
    # Reshape to profiles...
    x_gas   = np.reshape(x_gas,(ns,nlay,nx)) 
    lwp     = np.reshape(lwp,  (ns,nlay,1))    
    iwp     = np.reshape(iwp,  (ns,nlay,1))
    rsu     = np.reshape(rsu,  (ns,nlev))
    rsd     = np.reshape(rsd,  (ns,nlev))
    
    rsu_raw = np.copy(rsu)
    rsd_raw = np.copy(rsd)
    
    # normalize downwelling flux by the boundary condition
    rsd0    = rsd[:,0]
    rsd     = rsd / np.repeat(rsd0.reshape(-1,1), nlev, axis=1)
    # remove rsd0 from array
    rsd     = rsd[:,1:]
    # extract and remove upwelling flux at surface, this will be computed 
    # explicitly, resulting in NN outputs with consistent dimensions to input (nlay)
    rsu0    = rsu[:,-1]
    rsu     = rsu[:,0:-1]
    rsu     = rsu / np.repeat(rsd0.reshape(-1,1), nlay, axis=1)

    rsu     = rsu.reshape((ns,nlay,1))
    rsd     = rsd.reshape((ns,nlay,1))

    # Mu0 and surface albedo are also required as inputs
    # Don't know how to add constant (sequence-independent) variables,
    # so will add them as input to each sequence/level - unelegant but should work..
    mu0     = np.repeat(mu0.reshape(ns,1,1),nlay,axis=1)
    sfc_alb = np.repeat(sfc_alb.reshape(ns,1,1),nlay,axis=1)
    
    # Concatenate inputs and outputs...
    x       = np.concatenate((x_gas,lwp,iwp,mu0,sfc_alb),axis=2)
    y       = np.concatenate((rsd,rsu),axis=2)

    #print( "there are {} profiles in this dataset ({} experiments, {} columns)".format(nexp*ncol,nexp,ncol))
    
    dp = dat.variables['delta_pressure'][:,:,:].data       # (nexp,ncol, nlev)
    dp = np.reshape(dp,(ns,nlay))
    
    if return_coldry:
        coldry = get_col_dry(vmr_h2o,dp)
    
    dat.close()
    if return_p:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,dp,coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw,dp
    
    else:
        if return_coldry:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw, coldry
        else:
            return x,y,rsd0,rsu0,rsd_raw,rsu_raw

def load_data(file_name):

    x, _, rsd0, _, rsd, rsu, dp = \
        load_radscheme_rnn(file_name,  scale_p_h2o_o3 = True, return_p=True, return_coldry=False)
        
    xmax = np.array([3.20104980e+02, 1.15657101e+01, 4.46762830e-01, 5.68951890e-02,
        6.59545418e-04, 3.38450207e-07, 5.08802714e-06, 2.13372910e+02,
        1.96923096e+02, 1.00000000e+00, 1.00],dtype=np.float32)
    
    x = x / xmax

    nlay = x.shape[-2]

    y = np.concatenate((np.expand_dims(rsd,2), np.expand_dims(rsu,2)),axis=2)

    y = y / np.reshape(rsd0, (-1, 1, 1))

    rsd0_big = rsd0.reshape(-1,1).repeat(nlay+1,axis=1)

    # only one scalar input (albedo)
    x_m = x[:,:,0:-1]
    n_f = x_m.shape[-1]
    x_aux1 = x[:,0,-1:]   

    dt = xr.open_dataset(file_name)
    tmp_selection = dt.variables['is_valid_zenith_angle'].data.astype(int)
    tmp_selection = np.reshape(tmp_selection, (x_m.shape[0]))
    selection = tmp_selection.astype(bool)
    dt.close()

    return x_m[selection], x_aux1[selection], y[selection], dp[selection], rsd0_big[selection]


    
class non_sequential_access(Exception):
    "Raised when InputSequence is not sequentially accessed"
    pass
    


class InputSequence(tf.keras.utils.Sequence):

    def __reshuffle(self):
        self.e_shuffle = []   # Shuffle of examples within each month
        self.m_shuffle = np.array(list(range(12))) # shuffle of months
        random.shuffle(self.m_shuffle)
        acc = 0
        for i, m in enumerate(self.m_shuffle):
            c = self.n_data[m]
            acc += c  // self.batch_size
            self.n_batch_accumulated[i] = acc
            a = np.array(list(range(c)))
            random.shuffle(a)
            self.e_shuffle.append(a)

    def __shuffle(self, x_m, x_aux1, y, dp, rsd0_big):
        e = self.e_shuffle[self.i_file]
        self.x_m = tf.convert_to_tensor(x_m[e,:,:])
        self.x_aux1 = tf.convert_to_tensor(x_aux1[e,:])
        self.y = tf.convert_to_tensor(y[e,:,:])
        self.dp = tf.convert_to_tensor(dp[e,:])
        self.rsd0_big = tf.convert_to_tensor(rsd0_big[e,:])

    def __free_memory(self):
        del self.x_m
        del self.x_aux1
        del self.y 
        del self.dp
        del self.rsd0_big 

    def __init__(self, file_names, batch_size):
        self.file_names = file_names
        self.batch_size = batch_size
        self.n_data = []
        self.n_batch_accumulated = []
        acc = 0
        for f in file_names:
            dt = xr.open_dataset(f)
            #c = int(dt['mu0'].shape[0]* dt['mu0'].shape[1])
            c = int(np.sum(dt['is_valid_zenith_angle'].data))
            dt.close()
            self.n_data.append(c)
            acc += c // self.batch_size
            self.n_batch_accumulated.append(acc)

        self.i_file = 0
        print(f"Total number of examples = {np.sum(np.array(self.n_data))}")
        print(f"Number of valid examples per epoch = {acc * self.batch_size}")
        print(f"Number of valid batches = {acc}", flush=True)

    def __len__(self):
        return self.n_batch_accumulated[-1]

    def __getitem__(self, idx):
        # idx is a batch index

        # Assumes data is accessed sequentially
        # Verify that it is

        if idx > self.n_batch_accumulated[self.i_file]:
            print (f"idx = {idx}, max-idx = {self.n_batch_accumulated[self.i_file]}")
            raise non_sequential_access
        elif idx < 0:
            print (f"idx = {idx}")
            raise non_sequential_access
        else:
            if self.i_file > 0:
                if idx < self.n_batch_accumulated[self.i_file - 1]:
                    print (f"self.i_file = {self.i_file}")
                    print (f"idx = {idx}, min-idx = {self.n_batch_accumulated[self.i_file - 1]}")
                    raise non_sequential_access
                
        if idx == 0:
            if hasattr(self, 'x_m'):
                self.__free_memory()
                del self.e_shuffle
                del self.m_shuffle
            self.__reshuffle()
            self.i_file = 0
            x_m, x_aux1, y, dp, rsd0_big = load_data(self.file_names[self.m_shuffle[self.i_file]])
            self.__shuffle(x_m, x_aux1, y, dp, rsd0_big)  

        elif idx == self.n_batch_accumulated[self.i_file]:
            self.i_file = self.i_file + 1
            self.__free_memory()
            x_m, x_aux1, y, dp, rsd0_big = load_data(self.file_names[self.m_shuffle[self.i_file]])
            self.__shuffle(x_m, x_aux1, y, dp, rsd0_big)  

        if self.i_file == 0:
            i2 = idx
        else:
            i2 = idx - self.n_batch_accumulated[self.i_file - 1]

        i = i2 * self.batch_size
        j = (i2 + 1) * self.batch_size

        return ([self.x_m[i:j], self.x_aux1[i:j], self.y[i:j], self.dp[i:j], self.rsd0_big[i:j]], self.y[i:j])
        
    def on_epoch_end(self): 
        self.i_file = 0


def create_data_generator(file_names, scale_inputs=True):

    nlay=60
    nx_main = 10 # 5-gases + temp + pressure + mu + lwp + iwp

    signature = (tf.TensorSpec(shape=(nlay,nx_main),dtype=tf.float32),
                tf.TensorSpec(shape=(1,),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay+1,2),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay),dtype=tf.float32),
                tf.TensorSpec(shape=(nlay+1),dtype=tf.float32))
    
    def generator(file_names, scale_inputs=True):
        for f in file_names:
            print(f"Opening {f}")
            
            x_m, x_aux1, y, dp, rsd0_big = load_data(f)

            for i,_ in enumerate(x_m):
                yield (x_m[i], x_aux1[i], y[i], dp[i], rsd0_big[i])
    if True:
        return tf.data.Dataset.from_generator(generator,
                                          args=[file_names,True],
                                          output_signature=signature)
    else:
        return generator(file_names,scale_inputs)