import numpy as np
import random
import xarray as xr
import os

import torch
#import sklearn
from torch.utils.data import Dataset

def absorbed_flux_to_heating_rate(absorbed_flux, delta_pressure):

    # Note cp actually varies with temp and pressure: 
    # https://www.ohio.edu/mechanical/thermo/property_tables/air/air_Cp_Cv.html#:~:text=The%20nominal%20values%20used%20for,v%20%3D%200.718%20kJ%2Fkg.
    cp = 1004.0 # J K-1  kg-1 
    g = 9.80665
    flux_div_delta_pressure = absorbed_flux / delta_pressure
    return -(g/cp) * 24.0 * 3600.0 * flux_div_delta_pressure

def tensorize(np_ndarray, is_integer=False):
    if is_integer:
        t = torch.from_numpy(np_ndarray).int()
    else:
        t = torch.from_numpy(np_ndarray).float()
    return t

# is_clear_sky is a hack. The clear sky data is automatically included.
# However, when is_clear_sky the full sky data is swapped for the 
# clear sky data
def load_data(file, file_index, is_clear_sky):
    data = file

    temperature_pressure = data.variables['temp_pres_level'][:,:,:].data
    constituents = data.variables['constituents'][:,:,:].data
    delta_pressure = data.variables['delta_pressure'][:,:].data
    mu = data.variables['mu0'][:].data
    surface_albedo = data.variables['surface_albedo'][:].data

    flux_down_direct = data.variables['flux_down_direct'][:,:].data
    flux_down_diffuse = data.variables['flux_down_diffuse'][:,:].data
    flux_up_diffuse = data.variables['flux_up_diffuse'][:,:].data
    flux_down_direct_clear = data.variables['flux_down_direct_clear'][:,:].data
    flux_down_diffuse_clear = data.variables['flux_down_diffuse_clear'][:,:].data
    flux_up_diffuse_clear = data.variables['flux_up_diffuse_clear'][:,:].data

    selection = data.variables['is_valid_zenith_angle'].data.astype(int)

    if np.isnan(np.sum(temperature_pressure)):
        print(f"input {file_index} temperature_pressure contains NaN")
        os.abort()
    if np.isnan(np.sum(constituents)):
        print(f"input constituents contains NaN")
        for i in np.arange(constituents.shape[2]):
            if np.isnan(np.sum(constituents[:,:,i])):
                print(f"input {file_index} constituent={i} contains NaN")
                print(f"Indices of Nan = {np.argwhere(np.isnan(constituents[:,:,i]))}")
        os.abort()
    if np.isnan(np.sum(delta_pressure)):
        print(f"input {file_index} delta_pressure contains NaN")
        os.abort()
    if np.isnan(np.sum(mu)):
        print(f"input {file_index} mu contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_down_direct)):
        print(f"input {file_index} flux_down_direct contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_down_diffuse)):
        print(f"input {file_index} flux_down_diffuse contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_up_diffuse)):
        print(f"input {file_index} flux_up_diffuse contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_down_direct_clear)):
        print(f"input {file_index} flux_down_direct_clear contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_down_diffuse_clear)):
        print(f"input {file_index} flux_down_diffuse_clear contains NaN")
        os.abort()
    if np.isnan(np.sum(flux_up_diffuse_clear)):
        print(f"input {file_index} flux_up_diffuse_clear contains NaN")
        os.abort()
    
    if np.isnan(np.sum(selection)):
        print(f"input {file_index} is_valid_zenith_angle contains NaN")
        os.abort()

    # Select daylight examples only
    selection = selection.astype(bool)
    temperature_pressure = temperature_pressure[selection,:,:]
    constituents = constituents[selection,:,:]
    delta_pressure = delta_pressure[selection,:]
    mu = mu[selection]
    surface_albedo = surface_albedo[selection]
    # fluxes are "selected" after they are concatenated in a single tensor

    # Scale temperature and pressure approx between -1 and 1
    t_p_mean = np.array([248.6, 35043.8],dtype=np.float32)
    t_p_min = np.array([176.0, 0.0],dtype=np.float32)
    t_p_max = np.array([320.10498, 105420.29],dtype=np.float32)

    t_p_mean = t_p_mean.reshape((1, 1, -1))
    t_p_max = t_p_max.reshape((1, 1, -1))
    t_p_min = t_p_min.reshape((1, 1, -1))

    temperature_pressure = (temperature_pressure - t_p_mean)/ (t_p_max - t_p_min)

    # Scale remaining variables
    lw_max = 2.1337292e+02  #10X max
    iw_max = 1.9692309e+02  #10X max
    h2o_max = 4.84642649
    o3_max = 8.97450472e-04
    co2_max = 2.55393449e-01
    o2_max = 95.89
    n2o_max = 2.08013207e-04
    ch4_max = 4.39629856e-04
    #co_max = 9.5959631e-04

    constituents_norm = np.array([lw_max, iw_max, h2o_max,o3_max,co2_max,o2_max, n2o_max, ch4_max],dtype=np.float32)
    constituents = constituents / constituents_norm


    shape = flux_down_direct.shape
    flux_down_direct = flux_down_direct.reshape((shape[0], shape[1], 1))
    flux_down_diffuse = flux_down_diffuse.reshape((shape[0], shape[1], 1))
    flux_up_diffuse = flux_up_diffuse.reshape((shape[0], shape[1], 1))
    flux_down_direct_clear = flux_down_direct_clear.reshape((shape[0], shape[1], 1))
    flux_down_diffuse_clear = flux_down_diffuse_clear.reshape((shape[0], shape[1], 1))
    flux_up_diffuse_clear = flux_up_diffuse_clear.reshape((shape[0], shape[1], 1))

    if is_clear_sky:
        # Really really bad hack to swap in clear
        # sky data for full sky data. Done to quickly
        # test accuracy on clear sky data.
        # Should really
        # use the last set of inputs for clear sky

        # setting liquid water and ice water to zero
        constituents[:,:,0] = 0.0
        constituents[:,:,1] = 0.0
        y = np.concatenate((flux_down_direct_clear, flux_down_diffuse_clear, flux_up_diffuse_clear, flux_down_direct_clear, flux_down_diffuse_clear, flux_up_diffuse_clear), axis=2)
        y = y[selection,:,:]
    else:
        y = np.concatenate((flux_down_direct, flux_down_diffuse, flux_up_diffuse, flux_down_direct_clear, flux_down_diffuse_clear, flux_up_diffuse_clear), axis=2)
        y = y[selection,:,:]
        
    x_layers = np.concatenate((temperature_pressure, constituents), axis=2)


    mu = mu.reshape((-1,1))
    surface_albedo = surface_albedo.reshape((-1,1))
    x_surface = np.concatenate((mu, surface_albedo), axis=1)

    # Bad hack to embed geographic information in data
    n_sites = 5120
    n_time = selection.shape[0] / n_sites
    if selection.shape[0] % n_sites != 0:
        raise Exception("Number of Samples is not a multiple of the number of sites")
        os.abort()

    t1 = np.arange(n_sites, dtype=np.int32)
    t2 = [t1 for i in np.arange(n_time)]
    sites = np.concatenate(t2)
    sites = sites[selection]

    return tensorize(x_layers), tensorize(x_surface), tensorize(delta_pressure), tensorize(y), tensorize(sites, is_integer=True)


class RTDataSet(Dataset):
    # With shuffle of data files, too
    def __free_memory(self):
        del self.x_layers
        del self.x_surface
        del self.delta_pressure
        del self.y
        del self.sites

    def __reshuffle(self):
        # shuffle of months
        self.m_shuf = np.array(list(range(12)))
        random.shuffle(self.m_shuf)
        self.e_shuf = []
        acc = 0
        for i, m in enumerate(self.m_shuf):
            c = self.n_data[m]
            acc += c 
            self.n_data_accumulated[i] = acc
            # shuffle of examples
            a = np.array(list(np.arange(c)))
            random.shuffle(a)
            self.e_shuf.append(a)

    def __init__(self, input_files, is_clear_sky):
        self.dt = [xr.open_dataset(f) for f in input_files]
        self.n_data_accumulated = []
        self.n_data = []
        self.last_index = 0
        self.epoch_count = 0
        self.is_clear_sky = is_clear_sky
        acc = 0
        for d in self.dt:
            c = int(np.sum(d['is_valid_zenith_angle'].data))
            acc += c
            self.n_data_accumulated.append(acc)
            self.n_data.append(c)

        self.i_file = 0
        print(f"Number of valid examples = {acc}", flush=True)

    def __len__(self):
        return self.n_data_accumulated[-1]
    
    def __getitem__(self, idx):
        # Assumes data is accessed sequentially
        # Verify that it is

        assert idx <= self.n_data_accumulated[self.i_file], f"idx = {idx}, i_file = {self.i_file}, upper limit = {self.n_data_accumulated[self.i_file]}"

        assert idx >= 0, f"idx is negative: {idx}"

        if self.i_file > 0:
            assert idx >= self.n_data_accumulated[self.i_file - 1], f"idx = {idx}, i_file = {self.i_file}, lower limit = {self.n_data_accumulated[self.i_file - 1]}"

        if idx == 0:
            # Starting new epoch
            if hasattr(self, 'x_layers'):
                #self.__free_memory()
                #del self.e_shuf
                #del self.m_shuf
                pass
            self.epoch_count += 1
            self.i_file = 0
            self.__reshuffle()
            data = load_data(self.dt[self.m_shuf[self.i_file]], self.m_shuf[self.i_file], is_clear_sky=self.is_clear_sky)
            self.x_layers, self.x_surface, self.delta_pressure, self.y, self.sites = data

        elif idx == self.n_data_accumulated[self.i_file]:
            # Starting new file of data
            self.i_file = self.i_file + 1
            self.__free_memory()
            data = load_data(self.dt[self.m_shuf[self.i_file]], self.m_shuf[self.i_file], is_clear_sky=self.is_clear_sky)
            self.x_layers, self.x_surface, self.delta_pressure, self.y, self.sites = data

        assert self.x_layers.shape[0] == self.e_shuf[self.i_file].shape[0], f"len of x_layers = {self.x_layers.shape[0]}, len of shuff = {self.e_shuf[self.i_file].shape[0]}"

        self.last_index = idx
        if self.i_file == 0:
            ii = self.e_shuf[self.i_file][idx]
        else:
            ii = self.e_shuf[self.i_file][idx - self.n_data_accumulated[self.i_file - 1]]

        if True:
            if idx == self.n_data_accumulated[-1] - 1:
                # End of epoch
                # Note: self.i_file seems to be automatically reset to zero
                # at the end of an epoch. Can't figure out why.
                # Do it explicitly here, anyway
                self.i_file = 0
                
        return self.x_layers[ii], self.x_surface[ii], self.delta_pressure[ii], self.y[ii], self.sites[ii]

    
if __name__ == "__main__":
    # Testing
    if False:
        batch_size = 2048

        year = '2008'
        train_input_dir = f"/data-T1/hws/CAMS/processed_data/training/{year}/"
        validation_input_dir = f"/data-T1/hws/CAMS/processed_data/validation/{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        train_input_files = [f'{train_input_dir}nn_input-training-{year}-{month}.nc' for month in months]

        train_dataset = RTDataSet(train_input_files)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, 
                                                    shuffle=False,
                                                            num_workers=1)
        features = next(iter(train_dataloader))
        print(f"Number of features = {len(features)}")
        print (f"Shape of features[0] = {features[0].shape}")

    elif False:
        a = np.array(list(range(4)))
        print(f"a = {a}")
        random.shuffle(a)
        print(f'shuffled a = {a}')
    else:

        year = '2008'
        train_input_dir = f"/data-T1/hws/CAMS/processed_data/training/{year}/"
        validation_input_dir = f"/data-T1/hws/CAMS/processed_data/validation/{year}/"
        months = [str(m).zfill(2) for m in range(1,13)]
        train_input_files = [f'{train_input_dir}nn_input_sw-training-{year}-{month}.nc' for month in months]
        dt = xr.open_dataset(train_input_files[0])
        x = load_data(dt, 1, is_clear_sky=False)


