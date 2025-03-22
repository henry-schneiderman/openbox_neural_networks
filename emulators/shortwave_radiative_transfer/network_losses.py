import torch
import data_generation

def total_rmse(reference, pred):
    loss = torch.sqrt(torch.mean(torch.square(reference - pred),
                                 dim=(0,1,), keepdim=False))
    return loss

def total_bias(reference, pred):
    loss = torch.mean(reference - pred,dim=(0,1,), keepdim=False)
    return loss

def total_mae(reference, pred):
    loss = torch.mean(torch.abs(reference - pred),
                                  dim=(0,1,), keepdim=False)
    return loss

def layered_rmse(reference, pred):
    loss = torch.sqrt(torch.mean(torch.square(reference - pred),
                                 dim=(0,), keepdim=False))
    return loss

def layered_bias(reference, pred):
    loss = torch.mean(reference - pred,dim=(0,), keepdim=False)
    return loss

def layered_mae(reference, pred):
    loss = torch.mean(torch.abs(reference - pred),
                                  dim=(0,), keepdim=False)
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

def direct_extinction(flux_down_true, flux_down_pred,
                      delta_pressure, metric_function):
    
    flux_absorbed_true = (flux_down_true[:,:-1] - flux_down_true[:,1:])

    flux_absorbed_pred = (flux_down_pred[:,:-1] - flux_down_pred[:,1:])
    
    extinction_true = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_true, delta_pressure)
    extinction_pred = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_pred, delta_pressure)
    loss = metric_function(extinction_true, extinction_pred)

    return loss

def heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred,
                 delta_pressure, metric_function):
    
    flux_absorbed_true = (flux_down_true[:,:-1] - flux_down_true[:,1:] + 
                          flux_up_true[:,1:] - flux_up_true[:,:-1])
                            
    flux_absorbed_pred = (flux_down_pred[:,:-1] - flux_down_pred[:,1:] + 
                          flux_up_pred[:,1:] - flux_up_pred[:,:-1])
    
    heat_true = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_true, delta_pressure)
    heat_pred = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_pred, delta_pressure)
    
    loss = metric_function(heat_true, heat_pred)

    return loss


def geographic_heating_rate(flux_down_true, flux_up_true, 
                            flux_down_pred, flux_up_pred, 
                            delta_pressure, sites, number_of_sites, 
                            metric_function):
        
    flux_absorbed_true = (flux_down_true[:,:-1] - flux_down_true[:,1:] + 
                          flux_up_true[:,1:] - flux_up_true[:,:-1])
                            
    flux_absorbed_pred = (flux_down_pred[:,:-1] - flux_down_pred[:,1:] + 
                          flux_up_pred[:,1:] - flux_up_pred[:,:-1])
    
    heat_true = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_true, delta_pressure)
    
    heat_pred = data_generation.absorbed_flux_to_heating_rate(
        flux_absorbed_pred, delta_pressure)

    loss, count = metric_function(heat_true, heat_pred, sites, number_of_sites)

    return loss, count

def direct_extinction_maker(metric_function):
    def direct_extinction_wrapper(data, y_pred, loss_weights):
        _, _, delta_pressure, y_true, _, = data
        (flux_down_direct_pred, _, _, _) = y_pred
        flux_down_direct_true = y_true[:,:,0]

        loss = direct_extinction(flux_down_direct_true, flux_down_direct_pred, delta_pressure, metric_function)
        
        return loss
    return direct_extinction_wrapper

def loss_geographic_heating_rate_maker(metric_function, number_of_sites):
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

        hr_loss, hr_count = geographic_heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure, sites, number_of_sites, metric_function)
        
        return hr_loss, hr_count
    return loss_geographic_heating_rate_wrapper

def heating_rate_maker(metric_function):
    def heating_rate_wrapper(data, y_pred, loss_weights):
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

        loss = heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure, metric_function)
        
        return loss
    return heating_rate_wrapper

def diffuse_heating_rate_maker(metric_function):
    def diffuse_heating_rate_wrapper(data, y_pred, loss_weights):
        _, _, delta_pressure, y_true, _, = data
        (_, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred

        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        hr_loss = heating_rate(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, delta_pressure, metric_function)
        
        return hr_loss
    return diffuse_heating_rate_wrapper


def geographic_flux_maker(metric_function, number_of_sites, is_down):
    def geographic_flux_wrapper(data, y_pred, loss_weights):
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
            flux_error, flux_count = metric_function(flux_down_true, flux_down_pred, sites, number_of_sites)
        else:
            flux_error, flux_count = metric_function(flux_up_true, flux_up_pred, sites, number_of_sites)
        
        return flux_error, flux_count
    return geographic_flux_wrapper




def loss_flux(flux_down_true, flux_up_true, 
              flux_down_pred, flux_up_pred, 
              metric_function):  

    flux_pred = torch.concat((flux_down_pred,flux_up_pred),dim=1)
    flux_true = torch.concat((flux_down_true,flux_up_true),dim=1)

    loss = metric_function(flux_pred,flux_true)
    return loss


def direct_flux_maker (metric_function):
    def direct_flux_wrapper(data, y_pred, loss_weights):
        _, _, _, y_true, _ = data
        (flux_down_direct_pred, _, _, _) = y_pred
        flux_down_direct_true = y_true[:,:,0]

        loss = metric_function(flux_down_direct_true, flux_down_direct_pred)
        return loss
    return direct_flux_wrapper

def diffuse_flux_maker (metric_function):
    def diffuse_flux_wrapper(data, y_pred, loss_weights):
        _, _, _, y_true, _ = data
        (_, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred
        #(_, flux_down_diffuse_true, flux_up_diffuse_true, _, _, _) = y_true
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        loss = loss_flux(flux_down_diffuse_true, flux_up_diffuse_true,flux_down_diffuse_pred, flux_up_diffuse_pred, metric_function)
        return loss
    return diffuse_flux_wrapper

def flux_maker (metric_function):
    def flux_wrapper(data, y_pred, loss_weights):
        _, _, _, y_true, _ = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred

        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        loss = loss_flux(flux_down_true, flux_up_true, 
                         flux_down_pred, flux_up_pred, 
                         metric_function)
        return loss
    return flux_wrapper

def directional_flux_maker(metric_function, is_down):
    def flux_wrapper(data, y_pred, loss_weights):
        _, _, _, y_true, _ = data
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred

        flux_down_direct_true = y_true[:,:,0]
        flux_down_diffuse_true = y_true[:,:,1]
        flux_up_diffuse_true = y_true[:,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        if is_down:
            loss = metric_function(flux_down_true, flux_down_pred)
        else:
            loss = metric_function(flux_up_true, flux_up_pred)
        
        return loss
    return flux_wrapper

direct_extinction_rmse = direct_extinction_maker(total_rmse)
diffuse_heating_rate_rmse = diffuse_heating_rate_maker(total_rmse)

heating_rate_bias = heating_rate_maker(total_bias)
heating_rate_rmse = heating_rate_maker(total_rmse)
layered_heating_rate_bias = heating_rate_maker(layered_bias)
layered_heating_rate_rmse = heating_rate_maker(layered_rmse)

flux_bias = flux_maker(total_bias)
flux_rmse = flux_maker(total_rmse)
direct_flux_bias = direct_flux_maker(total_bias)
direct_flux_rmse = direct_flux_maker(total_rmse)
diffuse_flux_bias = diffuse_flux_maker(total_bias)
diffuse_flux_rmse = diffuse_flux_maker(total_rmse)
upwelling_flux_bias = directional_flux_maker(total_bias, is_down=False)
upwelling_flux_rmse = directional_flux_maker(total_rmse, is_down=False)
downwelling_flux_bias = directional_flux_maker(total_bias, is_down=True)
downwelling_flux_rmse = directional_flux_maker(total_rmse, is_down=True)
layered_upwelling_flux_bias = directional_flux_maker(layered_bias, is_down=False)
layered_upwelling_flux_rmse = directional_flux_maker(layered_rmse, is_down=False)
layered_downwelling_flux_bias = directional_flux_maker(layered_bias, is_down=True)
layered_downwelling_flux_rmse = directional_flux_maker(layered_rmse, is_down=True)
layered_flux_bias = flux_maker(layered_bias)
layered_flux_rmse = flux_maker(layered_rmse)

def mu_selector_flux_maker(mu_threshold, metric_function):
    def loss_function(data, y_pred, loss_weights):
        _, x_surface, _, y_true, _ = data
        mu = x_surface[:,0]
        selection = mu < mu_threshold
        (flux_down_direct_pred, flux_down_diffuse_pred, flux_up_diffuse_pred, _) = y_pred

        flux_down_direct_pred = flux_down_direct_pred[selection]
        flux_down_diffuse_pred = flux_down_diffuse_pred[selection]
        flux_up_diffuse_pred = flux_up_diffuse_pred[selection]

        flux_down_direct_true = y_true[selection,:,0]
        flux_down_diffuse_true = y_true[selection,:,1]
        flux_up_diffuse_true = y_true[selection,:,2]

        flux_down_true = flux_down_direct_true + flux_down_diffuse_true
        flux_up_true = flux_up_diffuse_true

        flux_down_pred = flux_down_direct_pred + flux_down_diffuse_pred
        flux_up_pred = flux_up_diffuse_pred

        loss = loss_flux(flux_down_true, flux_up_true, flux_down_pred, flux_up_pred, metric_function)
        return loss
    return loss_function


def openbox_rmse(data, y_pred, loss_weights):

    loss_direct_flux = direct_flux_rmse(data, y_pred, loss_weights)    
    loss_diffuse_flux = diffuse_flux_rmse(data, y_pred, loss_weights)

    loss_direct_extinction = direct_extinction_rmse(data, y_pred, loss_weights)
    loss_diffuse_heating_rate = diffuse_heating_rate_rmse(data, y_pred, loss_weights)

    w1 = loss_weights[0]
    w2 = loss_weights[1]
    w3 = loss_weights[2]
    w4 = loss_weights[3]

    loss = (1.0 / (w1 + w2 + w3 + w4)) * (w1 * loss_direct_flux + w2 * loss_diffuse_flux + w3 * loss_direct_extinction + w4 * loss_diffuse_heating_rate)

    return loss