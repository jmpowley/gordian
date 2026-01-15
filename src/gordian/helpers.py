import numpy as np

import jax

from sedpy.observate import load_filters

from pysersic.rendering import HybridRenderer

from .loading import load_cube_data
from .config import BandConfig, CubeConfig

def return_filters(band_config):
    """Returns filters fit by pipeline"""

    filters = []
    for filter_kwargs in band_config.values():
        filter = filter_kwargs.filter
        filters.append(filter)

    return filters

def return_wv_to_save(band_config, cube_config, fit_config):
    """Returns the wavelength vector used to produce a PySersic multi-band fit"""

    # Use cube wavelength range
    if fit_config.use_cube_wave:
        cube_config = CubeConfig(**cube_config)
        wave, _, _ = load_cube_data(**cube_config)
        wv_to_save = wave
    
    # Use grid of value in range of effective wavelengths
    else:
        filters = return_filters(band_config)
        sedpy_filters = [load_filters(["jwst_" + filter])[0] for filter in filters]
        waveffs = [filter.wave_effective / 1e4 for filter in sedpy_filters] # TODO: Change to conversions
        waveffs = np.asarray(waveffs)
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    
    # Optionally, invert wavelengths
    if fit_config.invert_wave:
        wv_to_save = 1 / wv_to_save

    return wv_to_save

def return_linked_param_at_wv(results_dict, param, mode, return_std : bool):

    # Load median, standard deviation and MAP
    med_at_wv, std_at_wv = results_dict["summary"][f"{param}_at_wv"]
    map_at_wv = results_dict["map"][f"{param}_at_wv"]

    # return median or MAP, optionally with standard deviation
    if mode == "median":
        if return_std:
            return med_at_wv, std_at_wv
        else:
            return med_at_wv
    elif mode == "MAP":
        if return_std:
            return map_at_wv, std_at_wv
        else:
            return map_at_wv
    
def return_linked_param_at_filter(results_dict, param, filter, mode, return_std : bool):

    # Load median, standard deviation and MAP
    med_at_filter, std_at_filter = results_dict["summary"][f"{param}_{filter}"]
    map_at_filter = results_dict["map"][f"{param}_{filter}"]

    print("filter:", filter)
    print(f"{param} MAP:", np.asarray(map_at_filter))
    print(f"{param} median:", np.asarray(med_at_filter))
    print(f"{param} std:", np.asarray(std_at_filter))

    print(results_dict["map"])

    # return median or MAP, optionally with standard deviation
    if mode == "median":
        if return_std:
            return med_at_filter, std_at_filter
        else:
            return med_at_filter
    elif mode == "MAP":
        if return_std:
            return map_at_filter, std_at_filter
        else:
            return map_at_filter
    else:
        raise Exception(f"Error: 'mode' argument {mode} not valid")

def return_const_param(results_dict, param, mode, return_std : bool):

    # Load median and standard deviation
    med, std = results_dict["summary"][f'{param}']
    map = results_dict["map"][f"{param}"]

    print("param:", param)
    print("MAP;", map)
    print("median:", med)

    # return median or MAP, optionally with standard deviation
    if mode == "median":
        if return_std:
            return med, std
        else:
            return med
    elif mode == "MAP":
        if return_std:
            return map, std
        else:
            return map
    else:
        raise Exception(f"Error: 'mode' argument {mode} not valid")
    
def return_individual_fit_posteriors(tree, params, return_medians : bool, return_uncs : bool):

    posterior = tree['posterior']
    param_posts = {}
    param_meds = []
    param_uncs = []

    # Wrap single params in list
    if type(params) != list:
        params = [params]

    for p in params:
        param_post = posterior[p][1]  # use second output...
        param_med = np.nanpercentile(param_post, q=50)
        param_16 = np.nanpercentile(param_post, q=16)
        param_84 = np.nanpercentile(param_post, q=84)

        param_posts[p] = param_post
        param_meds.append(param_med)
        param_uncs.append([param_med-param_16, param_84-param_med])

    # Optionally return image and residual
    if return_medians:
        if return_uncs:
            return param_posts, param_meds, param_uncs
        else:
            return param_posts, param_meds
    else:
        if return_uncs:
            return param_posts, param_uncs
        else:
            return param_posts
        
def return_simultaneous_fit_params_at_filter(simultaneous_tree, params, filter, mode, return_list : bool, return_unc : bool = False):

    # Load variables from tree
    sim_results_dict = simultaneous_tree["results_dict"]
    fit_config = simultaneous_tree["fit_config"]
    linked_params = fit_config["linked_params"]
    const_params = fit_config["const_params"]

    # Wrap single params in list
    if type(params) != list:
        params = [params]

    # Load parameter medians/MAPs from results dict
    params_at_filters = {}
    param_uncs_at_filters = {}
    for param in params:
        if param in const_params:
            param, param_unc = return_const_param(sim_results_dict, param=param, mode=mode, return_std=True)
        elif param in linked_params:
            param, param_unc = return_linked_param_at_filter(sim_results_dict, param=param, filter=filter, mode=mode, return_std=True)
        else:
            raise Exception(f"Parameter {param} not in list {linked_params + const_params}")
        params_at_filters[param] = param
        param_uncs_at_filters[param] = param_unc

    # Optionally return uncertainty as well or convert return to lists
    if return_unc:
        if return_list:
            return list(params_at_filters.values()), list(param_uncs_at_filters.values())
        else:
            return params_at_filters, param_uncs_at_filters
    else:
        if return_list:
            return list(params_at_filters.values())
        else:
            return params_at_filters
    
def return_median_model_from_individual_tree(tree, profile_type, use_image_flux, return_residual : bool, residual_type : str = "standard"):

    # Load data
    input_data = tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    rms = np.asarray(input_data["rms"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Calculate medians from posterior samples
    posterior = tree["posterior"]
    param_medians = {key : np.nanmedian(val) for key, val in posterior.items()}

    # Create theta vector
    theta = param_medians.copy()
    # -- add flux
    if use_image_flux:
        flux = np.nansum(image[mask])  # apply mask
        theta["flux"] = flux

    # Render model
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    median_model = np.asarray(renderer.render_source(params=theta, profile_type=profile_type))

    # Create residual
    # -- standard
    if residual_type == "standard":
        residual = image - median_model
    elif residual_type == "normalised":
        residual = (image - median_model) / rms
    elif residual_type == "relative_data":
        residual = (image - median_model) / image
    elif residual_type == "relative_model":
        residual = (image - median_model) / median_model
    else:
        raise ValueError(f"Residual type {residual_type} is not a valid.")

    # Optionally return residual
    if return_residual:
        return median_model, residual
    else:
        return median_model
    
def return_mean_model_from_individual_tree(tree, profile_type, use_image_flux, return_residual : bool):

    # Load data
    input_data = tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Calculate medians from posterior samples
    posterior = tree["posterior"]
    param_means = {key : np.nanmean(val) for key, val in posterior.items()}

    # Create theta vector
    theta = param_means.copy()
    # -- add flux
    if use_image_flux:
        flux = np.nansum(image[mask])  # apply mask
        theta["flux"] = flux

    # Render model and residual
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    mean_model = np.asarray(renderer.render_source(params=theta, profile_type=profile_type))
    residual = image - mean_model

    # Optionally return residual
    if return_residual:
        return mean_model, residual
    else:
        return mean_model

def return_median_model_from_simultaneous_tree(simultaneous_tree, individual_tree, profile_type, filter, use_image_flux, return_residual : bool, residual_type : str = "standard"):

    # Load data
    input_data = individual_tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    rms = np.asarray(input_data["rms"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Extract joint fit variables
    sim_results_dict = simultaneous_tree["results_dict"]
    fit_config = simultaneous_tree["fit_config"]
    linked_params = fit_config["linked_params"]
    const_params = fit_config["const_params"]
    all_params = linked_params + const_params

    # Load parameter medians from joint fits
    param_medians = {}
    for param in all_params:
        if param in const_params:
            param_median = return_const_param(sim_results_dict, param=param, mode="median", return_std=False)
        if param in linked_params:
            param_median = return_linked_param_at_filter(sim_results_dict, param=param, filter=filter, mode="median", return_std=False)
        param_medians[param] = param_median

    # Create theta vector
    theta = param_medians.copy()
    # -- add flux
    if use_image_flux:
        flux = np.nansum(image[mask])  # apply mask
        theta["flux"] = flux

    # Render model
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    median_model = np.asarray(renderer.render_source(params=theta, profile_type=profile_type))

    # Create residual
    # -- standard
    if residual_type == "standard":
        residual = image - median_model
    elif residual_type == "normalised":
        residual = (image - median_model) / rms
    elif residual_type == "relative_data":
        residual = (image - median_model) / image
    elif residual_type == "relative_model":
        residual = (image - median_model) / median_model
    else:
        raise ValueError(f"Residual type {residual_type} is not a valid.")

    # Optionally return residual
    if return_residual:
        return median_model, residual
    else:
        return median_model
    
def return_MAP_model_from_simultaneous_tree(simultaneous_tree, individual_tree, filter, return_residual : bool, residual_type : str = "standard"):

    # Load data
    input_data = individual_tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    rms = np.asarray(input_data["rms"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Extract joint fit variables
    sim_results_dict = simultaneous_tree["results_dict"]
    filters = simultaneous_tree["filters"]
    sim_map_dict = sim_results_dict["map"]

    # Load model
    filter_idx = filters.index(filter)
    map_model = sim_map_dict["model"][filter_idx]
    
    # Create residual
    # -- standard
    if residual_type == "standard":
        residual = image - map_model
    elif residual_type == "normalised":
        residual = (image - map_model) / rms
    elif residual_type == "relative_data":
        residual = (image - map_model) / image
    elif residual_type == "relative_model":
        residual = (image - map_model) / map_model
    else:
        raise ValueError(f"Residual type {residual_type} is not a valid.")

    # Optionally return residual
    if return_residual:
        return map_model, residual
    else:
        return map_model

def return_MAP_from_fitter(fitter, rkey):

    # Find MAP
    rkey, rkey_map = jax.random.split(rkey, 2)  # use different random number key for each run
    map = fitter.find_MAP(rkey_map)

    # Convert from JAX arrays
    map = {key : np.asarray(val) for key, val in map.items()}

    return map

def return_summary_from_estimation(fitter, rkey, method):

    rkey, rkey_est = jax.random.split(rkey, 2)
    # -- MCMC sampling
    if method == "mcmc":
        fitter.sample(rkey=rkey_est)
        results = fitter.sampling_results
    # -- SVI
    elif method == "svi-flow":
        results = fitter.estimate_posterior(method=method, rkey=rkey_est)

    summary = results.retrieve_med_std()

    return summary