# Run with a conda environment using the packages in the package_list.txt
from netCDF4 import Dataset
import numpy as np

def reformat_data_from_openbox_to_rnn(
    mode,month,year,base_directory,format_name):
    """
    Reformats data from the format used by the openbox neural network
    to the format used by the RNN developed by Peter Ukkonen
    
    mode = 'training' | 'validation' | 'testing' 
    """
    g = 9.80665 #
    m_co2 = 44.011 #
    m_dry = 28.970  # ZAMD
    m_h2o = 18.0154 # ZAMW
    m_o2 = 31.999
    m_o3 = 47.9985
    m_n2o = 44.013 #
    m_ch4 = 16.043 #
    m_co = 28.010
    n_sites = 5120
    
    d = base_directory + f'{mode}/{year}/'  
    file_name_openbox = d + f'shortwave-{mode}-{year}-{month}.nc'
    file_name_rnn = d + f'shortwave-{mode}-{format_name}-{year}-{month}.nc'

    dt_openbox = Dataset(file_name_openbox,"r")
    dt_rnn = Dataset(file_name_rnn,"w")
    
    constituents = dt_openbox.variables['constituents'][:,:,:].data
    n_features = constituents.shape[2]
    
    rsu = dt_openbox.variables['flux_up_diffuse'][:,:].data
    shape = rsu.shape
    expt = np.int32(shape[0]/n_sites)
    
    dt_rnn.createDimension("site", n_sites)
    dt_rnn.createDimension("expt", expt)
    dt_rnn.createDimension("layer", shape[1]-1)
    dt_rnn.createDimension("level", shape[1])
    dt_rnn.createDimension("feature", n_features-1)

    delta_pressure = dt_openbox.variables["delta_pressure"][:,:].data
    delta_pressure = delta_pressure.reshape((expt, n_sites, shape[1]-1))
    var_delta_pressure = dt_rnn.createVariable("delta_pressure","f4",("expt","site","layer"))
    var_delta_pressure[:] = delta_pressure[:]
    
    total_mass = (delta_pressure / g) 
    total_mass = np.expand_dims(total_mass,axis=3)

    rsu = rsu.reshape((expt, n_sites, shape[1]))
    var_rsu = dt_rnn.createVariable("rsu","f4",("expt","site","level"))
    var_rsu.setncattr("long_name","upwelling shortwave flux")
    var_rsu[:]= rsu[:]

    rsd_direct = dt_openbox.variables['flux_down_direct'][:,:].data
    rsd_diffuse = dt_openbox.variables['flux_down_diffuse'][:,:].data
    rsd = rsd_direct + rsd_diffuse
    rsd = rsd.reshape((expt, n_sites, shape[1]))
    var_rsd = dt_rnn.createVariable("rsd","f4",("expt","site","level"))
    var_rsd.setncattr("long_name","downwelling shortwave flux")
    var_rsd[:]= rsd[:]

    rsd_direct = rsd_direct.reshape((expt, n_sites, shape[1]))
    var_rsd_dir = dt_rnn.createVariable("rsd_dir","f4",("expt","site","level"))
    var_rsd_dir.setncattr("long_name","downwelling direct shortwave flux")
    var_rsd_dir[:]= rsd_direct[:]
    
    # Masses of the constituents
    constituents = constituents.reshape((expt, n_sites, shape[1]-1,n_features))

    lwp = constituents[:,:,:,0]
    iwp = constituents[:,:,:,1]
    
    var_lwp = dt_rnn.createVariable("cloud_lwp","f4",("expt","site","layer"))
    var_lwp.setncattr("long_name","cloud liquid water path")
    var_lwp.setncattr("units","g/kg")
    var_lwp[:] = lwp[:]

    var_iwp = dt_rnn.createVariable("cloud_iwp","f4",("expt","site","layer"))
    var_iwp.setncattr("long_name","cloud liquid water path")
    var_iwp.setncattr("units","g/kg")
    var_iwp[:] = iwp[:]
    
    is_valid_zenith_angle = dt_openbox.variables['is_valid_zenith_angle'][:].data
    is_valid_zenith_angle = is_valid_zenith_angle.reshape((expt,n_sites))
    var_is_valid_zenith_angle = dt_rnn.createVariable("is_valid_zenith_angle","f4",("expt","site"))
    var_is_valid_zenith_angle.setncattr("long_name","True if zenith angle is less than 90 degrees")
    var_is_valid_zenith_angle[:] = is_valid_zenith_angle[:]
    
    mu0 = dt_openbox.variables['mu0'][:].data
    mu0 = mu0.reshape((expt,n_sites))
    var_mu0 = dt_rnn.createVariable("mu0","f4",("expt","site"))
    var_mu0.setncattr("long_name","cosine of solar zenith angle")
    var_mu0[:] = mu0[:]
    
    temp_pressure = dt_openbox.variables['temp_pres_level'][:,:,:].data
    temp_pressure = temp_pressure.reshape((expt,n_sites,shape[1]-1,2))
    
    # Constituents are originally masses
    # Compute mass mixing ratios: mass_of_constituent / mass_of_dry_air
    r = constituents[:,:,:,2:] / (total_mass - constituents[:,:,:,2:])
        
    # Transform to volume mixing ratios
    m_mass = [m_h2o/m_dry, m_o3/m_dry, m_co2/m_dry, m_o2/m_dry, m_n2o/m_dry, m_ch4/m_dry] 
    r = r / m_mass
    
    rrtmgp_sw_input = np.concatenate((temp_pressure,r[:,:,:,0:3], r[:,:,:,4:6]), axis=3)
    
    var_rrtmgp_sw_input = dt_rnn.createVariable("rrtmgp_sw_input","f4",("expt","site","layer","feature"))
    var_rrtmgp_sw_input.setncattr("long_name","Inputs for RRTMGP shortwave gas optics")
    var_rrtmgp_sw_input.setncattr("comments","Features: tlay play h2o o3 co2 n2o ch4")
    var_rrtmgp_sw_input[:] = rrtmgp_sw_input[:]
    
    surface_albedo = dt_openbox.variables['surface_albedo'][:].data
    surface_albedo = surface_albedo.reshape((expt,n_sites))
    var_surface_albedo = dt_rnn.createVariable("sfc_alb","f4",("expt","site"))
    var_surface_albedo.setncattr("long_name","surface albedo")
    var_surface_albedo[:] = surface_albedo[:]
    
    dt_openbox.close()
    dt_rnn.close()

    
if __name__ == "__main__":

    base_directory = f'/data-T1/hws/CAMS/processed_data/'
    months = [str(m).zfill(2) for m in range(1,13)]

    format_name = 'rnn-format'

    modes = ['training', 'validation']
    for mode in modes:
        for year in ['2008',]:
            for month in months:
                reformat_data_from_openbox_to_rnn(
                    mode = mode,
                    month = month,
                    year = year,
                    base_directory = base_directory,
                    format_name = format_name)
                print(f'Created: {format_name} {mode} {year} {month}')
                
    for year in ['2009','2015','2020',]:
        mode = 'testing'
        for month in months:
            reformat_data_from_openbox_to_rnn(
                mode = mode,
                month = month,
                year = year,
                base_directory = base_directory,
                format_name = format_name)
            print(f'Created: {format_name} {mode} {year} {month}')
    