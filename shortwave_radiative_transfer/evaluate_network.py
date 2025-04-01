"""
Test and Evaluate an Open Box Neural Network for Shortwave Radiative Transfer

evaluate_network() - Evaluates neural network on testing datasets
                    - generates various error metrics: bias, rmse
                    - Can breakdown error by atmospheric layer,
                    geography, cosine zenith angle, clear sky vs
                    full sky.
                    - can write internal data to file for further analysis

Author: Henry Schneiderman, henry@pittdata.com
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from netCDF4 import Dataset

import train_network as tn
import data_generation
import network_losses as nl


class FullNetInternals(nn.Module):
    """ Computes full radiative transfer (direct and diffuse radiation)
    for an atmospheric column """

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

    def reset_dropout(self, dropout_p):
        self.optical_depth_net.reset_dropout(dropout_p)
        self.scattering_net.reset_dropout(dropout_p)

    def forward(self, x):
        x_layers, x_surface, _, _, _, = x

        (temperature_pressure,
         constituents) = (x_layers[:, :, 0:2],
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
        tau = self.optical_depth_net((temperature_pressure,
                                      constituents))

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
                loss[i, :] += loss_fn(data, y_pred, loss_weights).numpy()

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
    shape2 = lwp.numpy().shape

    example = np.arange(shape[0])
    layer = np.arange(shape[1])

    # lwp = xr.DataArray(lwp, coords=[time,site,layer], dims=("time","site","layer"), name="lwp")

    # iwp = xr.DataArray(iwp, coords=[time,site,layer], dims=("time","site","layer"), name="iwp")

    # r = xr.DataArray(r, coords=[time,site,layer],dims=("time","site","layer"), name="r")

    mu_diffuse = mu_diffuse.numpy().flatten()

    mu_diffuse_n = np.arange(mu_diffuse.shape[0])
    mu_direct = mu_direct.numpy()
    # s1 = np.shape(mu_direct)
    # mu_direct = np.reshape(mu_direct, (s1[0], s1[1]*s1[2]))

    rs_direct = s_direct.numpy()
    rs_diffuse = s_diffuse.numpy()
    rr_toa = r_toa.numpy()
    rr_surface = r_surface.numpy()

    is_bad = np.isnan(rs_direct).any() or np.isnan(rs_diffuse).any()
    print(f"is bad = {is_bad}")

    ds = xr.Dataset(
        data_vars={
            "lwp": (["example", "layer"], lwp.numpy()),
            "iwp": (["example", "layer"], iwp.numpy()),
            "o3": (["example", "layer"], o3.numpy()),
            "mu_diffuse": (["mu_diffuse_n"], mu_diffuse),
            "mu_direct": (["example"], mu_direct),
            "s_direct": (["example", "layer"], rs_direct),
            "s_diffuse": (["example", "layer"], rs_diffuse),
            "r_toa": (["example"], rr_toa),
            "r_surface": (["example"], rr_surface),
            "t_direct": (["example", "layer"], t_direct.numpy()),
            "t_diffuse": (["example", "layer"], t_diffuse.numpy()),
            "h2o": (["example", "layer"], h2o.numpy()),
        },
        coords={
            "example": example,
            "layer": layer,
            "mu_diffuse_n": mu_diffuse_n,
        },
    )

    ds.to_netcdf(output_file_name)
    ds.close()


def evaluate_network():

    if True:
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
            print('__CUDA Device Name:', torch.cuda.get_device_name(0))
            print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(
                0).total_memory/1e9)
            print(f'Device capability = {torch.cuda.get_device_capability()}')
            use_cuda = True
        else:
            use_cuda = False

    model_dir = "/home/hws/src/openbox_neural_networks/shortwave_radiative_transfer/models/"
    model_id = "v2."
    name_prefix = 'openbox.shortwave.'
    n_epoch = 596  # Selected epoch
    batch_size = 1024
    n_channel = 42
    n_constituent = 8

    # Several options for computing loss.
    # At most, one of the following should be set true:
    # is_use_internals, is_geographic_loss, is_layered_loss,
    # is_clear_sky
    # If none are True, computes standard losses averaged over the
    # entire dataset

    # For writing internal data to netcdf file
    is_use_internals = False
    geographic_loss_file_name = model_dir + f"{name_prefix}{model_id}"

    # Computes losses wrt to geographic location
    is_geographic_loss = False
    # THIS SHOULD NOT BE HARDCODED!!! Should be read from file
    number_of_sites = 5120

    # Computes loss of each atmospheric layer individually
    is_layered_loss = False

    # Can only compute one error type at a time (heating rate, up flux,
    # down flux)
    is_flux = False  # only matters when is_layered_loss = True or is_geographic_loss
    is_down = False  # only matters when is_layered_loss = True or is_geographic_loss

    # Computes losses for clear sky data
    is_clear_sky = False

    # Select dataset
    if True:
        mode = "testing"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/testing/"
        years = ("2009", "2015", "2020")
    else:
        mode = "training"
        processed_data_dir = "/data-T1/hws/CAMS/processed_data/training/"
        years = ("2008", )

    if is_use_internals:
        model = FullNetInternals(
            n_channel, n_constituent, dropout_p=0, device=device)
    else:
        model = tn.FullNet(n_channel, n_constituent,
                           dropout_p=0, device=device)

    model = model.to(device=device)
    model_filename = model_dir + f"{name_prefix}{model_id}"

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

        loss_weights = tn.get_loss_weights(n_epoch)

        checkpoint = torch.load(model_filename + str(n_epoch).zfill(3),
                                map_location=torch.device(device))
        print(f"Loaded Model: epoch = {n_epoch}")
        model.load_state_dict(checkpoint['model_state_dict'])

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
                nl.flux_rmse, nl.direct_flux_rmse, nl.diffuse_flux_rmse,
                nl.flux_bias, nl.downwelling_flux_rmse, nl.upwelling_flux_rmse,
                nl.downwelling_flux_bias, nl.upwelling_flux_bias,
                nl.heating_rate_rmse, nl.direct_extinction_rmse,
                nl.diffuse_heating_rate_rmse,
                nl.heating_rate_bias,
                loss_flux_0_01, loss_flux_0_05, loss_flux_0_10)

            loss_names = (
                "Openbox RMSE",
                "Flux RMSE", "Direct Flux RMSE", "Diffuse Flux RMSE",
                "Flux Bias", "Flux Down RMSE", "Flux up RMSE",
                "Flux Down Bias", "Flux up Bias ",
                "Heating Rate RMSE", "Direct Extinction RMSE",
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
                _ = tn.test_loop(test_dataloader, model,
                                 loss_functions,
                                 loss_names,
                                 loss_weights, device)



if __name__ == "__main__":

    evaluate_network()
