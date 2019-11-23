import numpy as np
import torch

def normal(dimensions, mean=0., stddev=1.):
    """Torch tensor of samples from a normal distribution, with given shape.

    Attributes:
        dimensions (tuple): shape of the output tensor 
        mean (float, optional): mean of the sampled normal distribution
        stddev (float, optional): standard deviation of the normal distribution
    
    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix with the samples

    """

    return torch.from_numpy(stddev * np.random.randn(*dimensions) + mean)


def uniform(dimensions, mini=0., maxi=1.):
    """Torch tensor of samples from an uniform distribution, with given shape.

    Attributes:
        dimensions (tuple): shape of the output tensor 
        mini (float, optional): lower limit of the sampled uniform distribution
        maxi (float, optional): upper limit of the sampled uniform distribution
    
    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix with the samples

    """
    return torch.from_numpy(np.random.random(dimensions)*(maxi-mini) + mini)


def transit_model(transit_duration, contact_ratio, time):
    """Models a transit lightcurve for the given parameters and normalized time.

    Normalization means that the curve outside the transit is centered around 0 
    and the depth of the transit is set to -1. The transit is in the middle of
    the vector, with the "greatest transit" exactly in the center.
    
    It calculates the normalized light values for the given times.

    It can work with broadcasting.
    
    Attributes:
        transit_duration (float): normalized duration of the transit with 
            respect to the orbital period
        contact_ratio (float): ratio between ingress or egress and the transit
        time_view (array): time values for which the model will be evaluated

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix. Returns values for each
        time in the input, in order.

    """
    # Calculate the times of the contact points at each lightcurve, from 0 to 1
    contact_1 = 0.5 - transit_duration/2 - contact_ratio * transit_duration
    contact_2 = 0.5 - transit_duration/2
    contact_3 = 0.5 + transit_duration/2
    contact_4 = 0.5 + transit_duration/2 + contact_ratio * transit_duration
    
    # Calculate masks for each section of the light curve
    mask_ingress = (time > contact_1) & (time <= contact_2)
    mask_transit = (time > contact_2) & (time <= contact_3)
    mask_egress  = (time > contact_3) & (time <= contact_4)
    
    # Calculate normalized light values by section of the lightcurve
    ingress = torch.cos((time-contact_1)/(contact_2-contact_1)
                        *np.pi) * 0.5 - 0.5
    transit = -1.
    egress  = torch.cos((time-contact_3)/(contact_4-contact_3)
                        *np.pi) *-0.5 - 0.5
    
    # Sum all sections
    lightcurve = (ingress * mask_ingress +
                  transit * mask_transit +
                  egress  * mask_egress)

    return lightcurve


def create_transit_lightcurve(len_global_lightcurve, len_local_lightcurve,
                              transit_duration, contact_ratio,
                              local_ratio = 4.,
                              noise_power=0., time_view=(-1,)):
    """Creates a local and global view for a transit-like normalized lightcurve.

    Normalization means that the curve outside the transit is centered around 0 
    and the depth of the transit is set to -1. The transit is in the middle of
    the vector, with the "greatest transit" exactly in the center. The global 
    view shows the full period, and the local view zooms on the transit, with 
    a constant width for the transit and curve before and after. 
    
    Global and local views are concatenated in the same vector.

    It can work with broadcasting.
    
    Attributes:
        len_global_lightcurve (int): number of points in the global lightcurve
        len_local_lightcurve (int): number of points in the local lightcurve
        transit_duration (float): normalized duration of the transit with 
            respect to the orbital period
        contact_ratio (float): ratio between ingress or egress and the transit
        local_ratio (float, optional): ratio between transit including contact
            and rest of the curve that is represented in the local view
        noise_power (float, optional): variance of gaussian noise to add to the 
            lightcurve. No noise by default
        time_view (tuple, optional): description of dimension along which the
            lightcurve vector will be set. 1D by default

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix. In the time view, the 
        values corresponding to the global view go first, and are followed by
        the local view.

    """
    assert type(len_global_lightcurve) == int
    assert type(len_local_lightcurve) == int
    
    # Calculate times for the local window
    local_start = 0.5 - (transit_duration * (0.5+contact_ratio)) * local_ratio
    local_end   = 0.5 + (transit_duration * (0.5+contact_ratio)) * local_ratio

    # Normalized time tensor, from 0 to 1 inclusive
    global_time = torch.linspace(0., 1., len_global_lightcurve
                                 ).view(*time_view)
    local_time  = torch.linspace(0, 1, len_local_lightcurve
                                 ).view(*time_view
                                 ) * (local_end - local_start) + local_start
    
    # Apply the transit model
    global_lightcurve = transit_model(transit_duration, 
                                      contact_ratio, 
                                      global_time)
    
    local_lightcurve  = transit_model(transit_duration, 
                                      contact_ratio, 
                                      local_time)
    
    # Calculate random noise
    global_noise = normal(tuple(global_lightcurve.size()), 
                          stddev=(noise_power**0.5).numpy())
    
    local_noise  = normal(tuple(local_lightcurve .size()), 
                          stddev=(noise_power**0.5).numpy())

    return torch.cat((global_lightcurve + global_noise,
                      local_lightcurve + local_noise), 2)


def sample_transit_lightcurves(nof_lightcurves, 
                               len_global_lightcurve, len_local_lightcurve, 
                               transit_duration_range = (0.001, 0.01),
                               contact_ratio_range = (0.1, 1.0),
                               noise_power_range = (0.001, 0.01),
                               ):
    """Creates a series of transit-like normalized lightcurves.

    Normalization means that the curve outside the transit is centered around 0 
    and the depth of the transit is set to -1. The transit is in the middle of
    the vector, with the "greatest transit" exactly in the center.
    
    Attributes:
        nof_lightcurves (int): number of lightcurves that will be created, 
            which will be stacked along the first dimension of the tensor
        len_global_lightcurve (int): number of points in the global lightcurve, 
            which will be set along the third dimension of the tensor
        len_local_lightcurve (int): number of points in the local lightcurve, 
            which will be set along the third dimension of the tensor
        transit_duration_range (tuple): range of the uniform distribution from
            which transit durations will be sampled for each light curve
        contact_ratio_range (tuple): range of the uniform distribution from
            which contact ratios will be sampled for each light curve
        noise_power_range (tuple): range of the uniform distribution from
            which noise powers will be sampled for each light curve

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix of size (nof_lightcurves,
            1, len_lightcurves), with different light curves along the third
            dimension

    """
    assert type(nof_lightcurves) == int
    assert type(len_global_lightcurve) == int
    assert type(len_local_lightcurve) == int

    # Random distribution for the normalized transit duration, defined as the 
    # ratio of the time between second and third contacts and the period
    transit_duration = uniform((nof_lightcurves, 1, 1), *transit_duration_range)

    # Random distribution for the time between first and second contacts divided 
    # by the transit duration, dependent on the relative size of planet and star
    contact_ratio = uniform((nof_lightcurves, 1, 1), *contact_ratio_range)
    
    # Random distribution for the noise power, measured as variance, for the 
    # gaussian distributions from which it will be sampled
    noise_power = uniform((nof_lightcurves, 1, 1), *noise_power_range)

    lightcurves = create_transit_lightcurve(len_global_lightcurve, len_local_lightcurve, 
                                            transit_duration, 
                                            contact_ratio, 
                                            noise_power=noise_power,
                                            time_view=(1, 1, -1))

    return lightcurves


def binary_model(transit_duration, contact_ratio, time):
    """Models a binary lightcurve for the given parameters and normalized time.

    Normalization means that the curve outside the transit is centered around 0 
    and the depth of the transit is set to -1. The main eclipse is in the middle 
    of the vector, with the "greatest occultation" exactly in the center.
    
    It calculates the normalized light values for the given times.

    It can work with broadcasting.
    
    Attributes:
        transit_duration (float): normalized duration of the eclipse with 
            respect to the orbital period
        contact_ratio (float): ratio between ingress or egress and the transit
        time_view (array): time values for which the model will be evaluated

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix. Returns values for each
        time in the input, in order.

    """
    # Calculate the times of the contact points at each lightcurve, from 0 to 1
    contact_1 = 0.5 - transit_duration/2 - contact_ratio * transit_duration
    contact_2 = 0.5 - transit_duration/2
    contact_3 = 0.5 + transit_duration/2
    contact_4 = 0.5 + transit_duration/2 + contact_ratio * transit_duration
    
    # Parameter for the model
    contact_depth = contact_ratio * np.exp(-contact_ratio)*0.2-1
    
    # Calculate masks for each section of the light curve
    mask_ingress = (time > contact_1) & (time <= contact_2)
    mask_transit = (time > contact_2) & (time <= contact_3)
    mask_egress  = (time > contact_3) & (time <= contact_4)
    
    # Calculate normalized light values by section of the lightcurve
    ingress = (time-contact_1) / (contact_2-contact_1) * contact_depth
    transit = -(1+contact_depth)*torch.cos(
               (time-0.5)*np.pi/(contact_3-contact_2))+contact_depth
    egress  = contact_depth - (time-contact_3) / (
                               contact_4-contact_3) * contact_depth 

    # Sum all sections
    lightcurve = (ingress * mask_ingress +
                  transit * mask_transit +
                  egress  * mask_egress)

    return lightcurve


def create_binary_lightcurve(len_global_lightcurve, len_local_lightcurve,
                              transit_duration, contact_duration,
                              local_ratio = 4.,
                              noise_power=0., time_view=(-1,)):
    """Creates a local and global view for a binary eclipse-like normalized 
    lightcurve.

    Normalization means that the curve outside the eclipse is centered around 0 
    and the depth of the eclipse is set to -1. The main eclipse is in the 
    middle of the vector, with the "greatest occultation" exactly in the center. 
    
    The global view shows the full period, and the local view zooms on the 
    eclipse, with a constant width for the transit and curve before and after. 
    
    Global and local views are concatenated in the same vector.

    It can work with broadcasting.
    
    Attributes:
        len_global_lightcurve (int): number of points in the global lightcurve
        len_local_lightcurve (int): number of points in the local lightcurve
        transit_duration (float): normalized duration of the eclipse with 
            respect to the orbital period
        contact_duration (float): normalized duration of ingress or egress with
            respect to the orbital period
        local_ratio (float, optional): ratio between transit including contact
            and rest of the curve that is represented in the local view
        noise_power (float, optional): variance of gaussian noise to add to the 
            lightcurve. No noise by default
        time_view (tuple, optional): description of dimension along which the
            lightcurve vector will be set. 1D by default

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix. In the time view, the 
        values corresponding to the global view go first, and are followed by
        the local view.

    """
    assert type(len_global_lightcurve) == int
    assert type(len_local_lightcurve) == int
    
    # Parameters for the model
    contact_ratio = contact_duration/transit_duration
    
    # Calculate times for the local window
    local_start = 0.5 - (transit_duration * (0.5+contact_ratio)) * local_ratio
    local_end   = 0.5 + (transit_duration * (0.5+contact_ratio)) * local_ratio

    # Normalized time tensor, from 0 to 1 inclusive
    global_time = torch.linspace(0., 1., len_global_lightcurve
                                 ).view(*time_view)
    local_time  = torch.linspace(0, 1, len_local_lightcurve
                                 ).view(*time_view
                                 ) * (local_end - local_start) + local_start
    
    # Apply the transit model
    global_lightcurve = binary_model(transit_duration, 
                                     contact_ratio, 
                                     global_time)
    
    local_lightcurve  = binary_model(transit_duration, 
                                     contact_ratio, 
                                     local_time)
    
    # Calculate random noise
    global_noise = normal(tuple(global_lightcurve.size()), 
                          stddev=(noise_power**0.5).numpy())
    
    local_noise  = normal(tuple(local_lightcurve .size()), 
                          stddev=(noise_power**0.5).numpy())

    return torch.cat((global_lightcurve + global_noise,
                      local_lightcurve + local_noise), 2)


def sample_binary_lightcurves(nof_lightcurves, 
                              len_global_lightcurve, len_local_lightcurve, 
                              transit_duration_range = (0.001, 0.01),
                              contact_duration_range = (0.001, 0.01),
                              noise_power_range = (0.001, 0.01),
                              ):
    """Creates a binary eclipse-like normalized lightcurves.

    Normalization means that the curve outside the transit is centered around 0 
    and the depth of the transit is set to -1. The transit is in the middle of
    the vector, with the "greatest transit" exactly in the center.
    
    Attributes:
        nof_lightcurves (int): number of lightcurves that will be created, 
            which will be stacked along the first dimension of the tensor
        len_global_lightcurve (int): number of points in the global lightcurve, 
            which will be set along the third dimension of the tensor
        len_local_lightcurve (int): number of points in the local lightcurve, 
            which will be set along the third dimension of the tensor
        transit_duration_range (tuple): range of the uniform distribution from
            which transit durations will be sampled for each light curve
        contact_duration_range (tuple): range of the uniform distribution from
            which contact durations will be sampled for each light curve
        noise_power_range (tuple): range of the uniform distribution from
            which noise powers will be sampled for each light curve

    Returns:
        torch.Tensor: PyTorch multi-dimensional matrix of size (nof_lightcurves,
            1, len_lightcurves), with different light curves along the third
            dimension

    """
    assert type(nof_lightcurves) == int
    assert type(len_global_lightcurve) == int
    assert type(len_local_lightcurve) == int

    # Random distribution for the normalized transit duration, defined as the 
    # ratio of the time between second and third contacts and the period
    transit_duration = uniform((nof_lightcurves, 1, 1), *transit_duration_range)

    # Random distribution for the time between first and second contacts divided 
    # by the transit duration, dependent on the relative size of planet and star
    contact_duration = uniform((nof_lightcurves, 1, 1), *contact_duration_range)
    
    # Random distribution for the noise power, measured as variance, for the 
    # gaussian distributions from which it will be sampled
    noise_power = uniform((nof_lightcurves, 1, 1), *noise_power_range)

    lightcurves = create_binary_lightcurve(len_global_lightcurve, len_local_lightcurve,
                                           transit_duration,
                                           contact_duration, 
                                           noise_power=noise_power,
                                           time_view=(1, 1, -1))

    return lightcurves