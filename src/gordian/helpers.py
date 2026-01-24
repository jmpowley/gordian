import numpy as np

import jax

from sedpy.observate import load_filters

from pysersic.rendering import HybridRenderer

from .loading import load_cube_data
from .config import BandConfig, CubeConfig, ModelConfig, FitConfig, IOConfig

# ------------------------------------
# Return variables from configurations
# ------------------------------------
def return_filters(band_config_dict):
    """Returns filters fit by Gordian."""

    filters = []
    for band_config in band_config_dict.values():
        filter = band_config.filter
        filters.append(filter)

    return filters


def return_wv_to_save(band_config, cube_config, fit_config):
    """Returns the wavelength passed to PySersic multi-band fit."""

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


def convert_band_config_dicts(band_config_dicts_in):

    band_config_dicts_out = {}

    for band_key, band_config_dict in band_config_dicts_in.items():

        band_config = BandConfig(**band_config_dict)
        band_config_dicts_out[band_key] = band_config

    return band_config_dicts_out


def return_waveffs_from_simultaneous_tree(tree):

    fit_config = FitConfig(**tree["fit_config"])
    results_dict = tree["results_dict"]

    waveffs = np.array(results_dict["waveffs"])

    if fit_config.invert_wave:
        waveffs = 1 / waveffs

    return waveffs


def return_wv_to_save_from_simultaneous_tree(tree):

    fit_config = FitConfig(**tree["fit_config"])
    results_dict = tree["results_dict"]

    wv_to_save = np.array(results_dict["wv_to_save"])

    if fit_config.invert_wave:
        wv_to_save = 1 / wv_to_save

    return wv_to_save

# ---------------------
# Return fit parameters
# ---------------------

def return_linked_param_at_wv(results_dict, param, mode, return_std : bool):
    """Returns a linked parameter value from the simultaneous fit results dictionary at each wavelength."""

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
    """Returns a linked parameter value from the simultaneous fit results dictionary for a given filter."""

    # Load median, standard deviation and MAP
    med_at_filter, std_at_filter = results_dict["summary"][f"{param}_{filter}"]
    map_at_filter = results_dict["map"][f"{param}_{filter}"]

    # Return median
    if mode == "median":
        if return_std:
            return med_at_filter, std_at_filter
        else:
            return med_at_filter
    
    # Return MAP
    elif mode == "MAP":
        if return_std:
            return map_at_filter, std_at_filter
        else:
            return map_at_filter
    
    else:
        raise Exception(f"Invalid mode, got {mode}")


def return_const_param(results_dict, param, mode, return_std : bool):
    """Returns a constant parameter value from the simultaneous fit results dictionary."""

    # Load median and standard deviation
    med, std = results_dict["summary"][f'{param}']
    map = results_dict["map"][f"{param}"]

    # Return median
    if mode == "median":
        if return_std:
            return med, std
        else:
            return med
        
    # Return MAP
    elif mode == "MAP":
        if return_std:
            return map, std
        else:
            return map
    
    else:
        raise Exception(f"Invalid mode, got {mode}")

# ----------------------------
# Return fit params from trees
# ----------------------------

def return_individual_fit_param(individual_tree, param, mode: str, return_unc: bool):
    """Returns a parameter from the individual fit posterior distribution."""

    # NOTE: To be consistent with return_*_param* functions, return median and standard deviation
    # In future, consider return quantiles of posterior distribution instead

    # Extract parameter posterior
    posterior = individual_tree['posterior']
    param_post = posterior[param][1]  # use second run

    # Calculate value and uncertainties
    param_med = np.nanmedian(param_post)
    param_mean = np.nanmean(param_post)
    param_std = np.nanstd(param_post)
    param_16 = np.nanpercentile(param_post, q=16)
    param_84 = np.nanpercentile(param_post, q=84)

    # Return median
    if mode == "median":
        if return_unc:
            return param_med, param_std
        else:
            return param_med

    # Return mean
    elif mode == "mean":
        if return_unc:
            return param_mean, param_std
        else:
            return param_mean
        
    else:
        raise Exception(f"Invalid mode, got {mode}")

# def return_simultaneous_fit_params_at_filter(simultaneous_tree, params, filter, mode, return_list: bool, return_unc: bool = False):
#     """Returns values of all parameters from a simultaneous fit for a given filter"""

#     # Load variables from tree
#     sim_results_dict = simultaneous_tree["results_dict"]

#     fit_config = FitConfig(**simultaneous_tree["fit_config"])
#     linked_params = fit_config.linked_params
#     const_params = fit_config.const_params

#     # Wrap single params in list
#     if type(params) != list:
#         params = [params]

#     # Load parameter medians/MAPs from results dict
#     params_at_filter = {}
#     param_uncs_at_filter = {}
#     for param in params:
#         if param in const_params:
#             param, param_unc = return_const_param(sim_results_dict, param=param, mode=mode, return_std=True)
#         elif param in linked_params:
#             param, param_unc = return_linked_param_at_filter(sim_results_dict, param=param, filter=filter, mode=mode, return_std=True)
#         else:
#             raise Exception(f"Parameter {param} not in list {linked_params + const_params}")
#         params_at_filter[param] = param
#         param_uncs_at_filter[param] = param_unc

#     # Optionally return uncertainty as well or convert return to lists
#     if return_unc:
#         if return_list:
#             return list(params_at_filter.values()), list(param_uncs_at_filter.values())
#         else:
#             return params_at_filter, param_uncs_at_filter
#     else:
#         if return_list:
#             return list(params_at_filter.values())
#         else:
#             return params_at_filter
        
# ------------------------
# Return models from trees
# ------------------------

def return_model_from_individual_tree(tree, model_config: ModelConfig, mode: str, use_image_flux: bool = True):

    # Load data
    input_data = tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    rms = np.asarray(input_data["rms"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Load posterior samples
    posterior = tree["posterior"]

    # Use mean
    if mode == "mean":
        param_values = {key : np.nanmean(val) for key, val in posterior.items()}

    # Use median
    elif mode == "median":
        param_values = {key : np.nanmedian(val) for key, val in posterior.items()}

    # Create theta vector
    theta = param_values.copy()
    
    # Add flux
    if use_image_flux:
        flux = np.nansum(image[mask])  # apply mask
        theta["flux"] = flux

    # Render model
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    model = np.asarray(renderer.render_source(params=theta, profile_type=model_config.profile_type))

    return model

def return_model_from_simultaneous_tree(simultaneous_tree, individual_tree, filter: str, mode: str, use_image_flux: bool = True):
    """Return the model of a specified type (mean or median) from a simultaneous fit."""

    # Extract simultaneous fit variables
    sim_results_dict = simultaneous_tree["results_dict"]
    fit_config = FitConfig(**simultaneous_tree["fit_config"])
    model_config = ModelConfig(**simultaneous_tree["model_config"])

    # Load parameters
    linked_params = fit_config.linked_params
    const_params = fit_config["const_params"]
    all_params = linked_params + const_params

    # Load data from individual tree
    input_data = individual_tree["input_data"]
    image = np.asarray(input_data["image"])
    mask = np.asarray(input_data["mask"])
    rms = np.asarray(input_data["rms"])
    psf = np.asarray(input_data["psf"]).astype(np.float32)  # recast as numpy

    # Load parameter values from joint fits
    param_values = {}
    for param in all_params:
        if param in const_params:
            param_value = return_const_param(sim_results_dict, param=param, mode=mode, return_std=False)
        if param in linked_params:
            param_value = return_linked_param_at_filter(sim_results_dict, param=param, filter=filter, mode=mode, return_std=False)
        param_values[param] = param_value

    # Create theta vector
    theta = param_values.copy()
    
    # Add flux
    if use_image_flux:
        flux = np.nansum(image[mask])  # apply mask
        theta["flux"] = flux

    # Render model
    renderer = HybridRenderer(im_shape=image.shape, pixel_PSF=psf)
    model = np.asarray(renderer.render_source(params=theta, profile_type=model_config.profile_type))

    return model

def return_MAP_model_from_simultaneous_tree(simultaneous_tree, filter: str):
    """Return the MAP model from a simultaneous fit.
    
    Note: MAP returned from fitter differently to median/mean model, so call is different
    """

    # Extract joint fit variables
    sim_results_dict = simultaneous_tree["results_dict"]
    filters = simultaneous_tree["filters"]
    sim_map_dict = sim_results_dict["map"]

    # Load model from dict
    filter_idx = filters.index(filter)
    map_model = sim_map_dict["model"][filter_idx]
    
    return map_model


def return_model_residual(image: np.ndarray, model: np.ndarray, rms: np.ndarray = None, residual_type: str = "standard"):
    """Returns the residual from input data and model.
    
    Different kinds of residual available to specify.
    """

    # Standard residual
    if residual_type == "standard":
        residual = image - model

    # Error-normalised residual
    elif residual_type == "normalised":
        residual = (image - model) / rms

    # Residual compared to image
    elif residual_type == "relative_data":
        residual = (image - model) / image

    # Residual compared to model
    elif residual_type == "relative_model":
        residual = (image - model) / model
    
    else:
        raise ValueError(f"Invalid residual_type, got {residual_type}")
    
    return residual

# ------------------------------------------------
# Return estimators from fitter objects/posteriors
# ------------------------------------------------
def return_chains_summary_from_fitter(fitter, band_config_dict, rkey, method):

    # Sample posterior
    rkey, rkey_est = jax.random.split(rkey, 2)
    if method == "mcmc":
        # mcmc sampling
        fitter.sample(rkey=rkey_est)
        results = fitter.sampling_results
    elif method == "svi-flow":
        # svi
        results = fitter.estimate_posterior(method=method, rkey=rkey_est)

    # Create chains dict
    chain_dict = {}
    chains = results.get_chains()
    filters = return_filters(band_config_dict)

    # constant parameters
    for params in fitter.const_params:
        chain_dict[params] = chains[params].values

    # linked parameters
    for param in fitter.linked_params:
       for filter in filters:
           chain_dict[f"{param}_{filter}"] = chains[f"{param}_{filter}"].values
           chain_dict[f"{param}_at_wv"] = chains[f"{param}_at_wv"].values

    # unlinked params
    for param in fitter.unlinked_params:
        chain_dict[f"{param}_{filter}"] = chains[f"{param}_{filter}"].values

    # Create summary
    summary = results.retrieve_med_std()

    return chain_dict, summary


def return_MAP_from_fitter(fitter, rkey):

    # Find MAP
    rkey, rkey_map = jax.random.split(rkey, 2)  # use different random number key for each run
    map = fitter.find_MAP(rkey_map)

    # Convert from JAX arrays
    map = {key : np.asarray(val) for key, val in map.items()}

    return map