import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from sedpy.observate import load_filters

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

import numpyro.distributions as dist

from pysersic import FitSingle
from pysersic.multiband import FitMultiBandPoly, FitMultiBandBSpline
from pysersic.priors import autoprior, PySersicSourcePrior
from pysersic import loss

from .config import BandConfig, CubeConfig, ModelConfig, ModelConfig, IOConfig
from .loading import load_band_data, load_cube_data
from .plotting import make_plots
from .writing import save_individual_fit_result, save_simultaneous_fit_results
from .helpers import (
    return_linked_param_at_wv, 
    return_const_param, 
    return_MAP_from_fitter,
    return_chains_summary_from_fitter,
)


def set_priors(image: np.ndarray, mask: np.ndarray, model_config: ModelConfig):
    """Sets priors for model in PySersic.

    Parameters
    ----------
    image : np.ndarray
        Input image data
    mask : np.ndarray
        Input mask data
    model_config : ModelConfig
        Configuration for a model parameters

    Returns
    -------
    prior : pysersic.priors.PySersicSourcePrior
        Prior to be incorporated into PySersic fit
    """
    
    # Generate priors from image
    prior = autoprior(
        image=image, 
        profile_type=model_config.profile_type, 
        mask=mask, 
        sky_type=model_config.sky_type, 
    )

    # Set priors from config dict
    for prior_type, prior_type_dict in model_config.prior_dict.items():

        if prior_type == "uniform":
            for param, range in prior_type_dict.items():
                lo, hi = range
                prior.set_uniform_prior(param, lo, hi)

        if prior_type == "gaussian":
            for param, gauss in prior_type_dict.items():
                loc, std = gauss
                prior.set_gaussian_prior(param, loc, std)

        if prior_type == "fixed":
            for param, value in prior_type_dict.items():
                print(param, value)
                prior.set_custom_prior(param, dist.Delta(value))

    return prior


def fit_single(
        data: np.ndarray, 
        mask: np.ndarray, 
        rms: np.ndarray, 
        psf: np.ndarray, 
        prior, 
        loss_func: str, 
        method: str, 
        rkey, 
        verbose: bool,
        ):
    """Fits single image. Either a photometric band or a slice from an IFU cube."""

    if verbose:
        print("Starting fit...")

    # Map for available loss functions
    loss_map = {
        "student_t_loss": loss.student_t_loss,
        "student_t_loss_free_sys": loss.student_t_loss_free_sys,
        "gaussian_loss": loss.gaussian_loss,
        "cash_loss": loss.cash_loss,
    }
    loss_func = loss_map[loss_func]

    # Run fit
    fitter = FitSingle(data=data, rms=rms, mask=mask, psf=psf, prior=prior, loss_func=loss_func)

    # Sample posterior
    rkey, rkey_est = jax.random.split(rkey, 2)
    if method == "mcmc":
        fitter.sample(rkey=rkey_est)
        result = fitter.sampling_results
    elif method == "svi":
        result = fitter.estimate_posterior(rkey=rkey_est)
    
    if verbose:
        print("Result from fit:")
        if method == "mcmc":
            print(result.retrieve_param_quantiles(return_dataframe=True))

    return fitter, result


def fit_bands_independent(
    band_config: Dict[str, BandConfig],
    model_config: ModelConfig,
    fit_config: ModelConfig,
    io_config: IOConfig,
):
    """Fits set of photometric bands independently of each other.

    Parameters
    ----------
    band_config : Dict[str, BandConfig]
        Configuration for a single photometric band
    model_config : ModelConfig
        Configuration for a model parameters
    fit_config : FitConfig
        Configuration for a fitting
    io_config : IOConfig
        Configuration for a data input/output

    Returns
    -------
    _type_
        _description_
    """
    
    rkey = PRNGKey(fit_config.seed)

    # Loop over each band
    for band_name, band_config in band_config.items():

        filter = band_config.filter
        if io_config.verbose:
            print("------------------------------")
            print(f"Filter: {filter.upper()}")

        # Load cutout data
        image, mask, sig, psf = load_band_data(band_config)

        # Set priors
        prior = set_priors(image=image, mask=mask, model_config=model_config)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_single(
            data=image, 
            mask=mask, 
            rms=sig, 
            psf=psf, 
            prior=prior, 
            loss_func=fit_config.loss_func, 
            method=fit_config.method, 
            rkey=rkey_fit, 
            verbose=io_config.verbose
        )

        # Make plots
        make_plots(
            fitter, image, mask, sig, psf, 
            profile_type=model_config.profile_type, 
            method=fit_config.method, 
            filter=filter, 
            fig_dir=io_config.fig_dir, 
            )

        # Save results
        save_individual_fit_result(result=result, 
                                   model_config=model_config, 
                                   fit_config=fit_config, 
                                   io_config=io_config, 
                                   )

def fit_bands_simultaneous(
    band_config_dict: Dict[str, BandConfig],
    cube_config: Optional[CubeConfig],
    model_config: ModelConfig,
    fit_config: ModelConfig,
    io_config: IOConfig,
):
    """Fit bands simultaneously with wavelength-dependent parameters"""

    # Initialise key
    rkey = PRNGKey(fit_config.seed)

    # Initialise lists/dicts
    filters = []
    waveffs = []
    im_dict = {}
    mask_dict = {}
    rms_dict = {}
    psf_dict = {}
    ind_fitter_dict = {}
    ind_results_dict = {}
    sim_results_dict = {}

    # Loop over each band
    for band_name, band_config in band_config_dict.items():

        # Load filter
        filter = band_config.filter
        sedpy_filter = load_filters(["jwst_" + filter])[0]
        
        if io_config.verbose:
            print("------------------------------")
            print(f"Filter: {filter.upper()}")

        # Load cutout data
        image, mask, sig, psf = load_band_data(band_config)

        # Create PySersic priors
        prior = set_priors(image=image, mask=mask, model_config=model_config)
        if io_config.verbose:
            print(prior)

        # Fit band
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_single(
            data=image, 
            mask=mask, 
            rms=sig, 
            psf=psf, 
            prior=prior, 
            loss_func=fit_config.loss_func, 
            method=fit_config.method, 
            rkey=rkey_fit, 
            verbose=io_config.verbose,
        )

        # Save results
        out_name = f"{model_config.profile_type}_{fit_config.method}_fit_{filter}.asdf"
        out_path = os.path.join(io_config.out_dir, out_name)
        result.save_result(out_path)

        # Make plots
        make_plots(
            fitter, image, mask, sig, psf, 
            profile_type=model_config.profile_type, 
            method=fit_config.method, 
            filter=filter, 
            fig_dir=io_config.fig_dir
        )

        # Save to lists/dicts
        filters.append(filter)
        waveffs.append(sedpy_filter.wave_effective / 1e4)
        im_dict[filter] = image
        mask_dict[filter] = mask
        rms_dict[filter] = sig
        psf_dict[filter] = psf
        ind_fitter_dict[filter] = fitter
        ind_results_dict[filter] = result

    # Create wavelength vectors to save
    waveffs = np.asarray(waveffs)
    if fit_config.use_cube_wave and cube_config is not None:
        wave, _, _, _ = load_cube_data(cube_config)
        wv_to_save = wave
    else:
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    
    # Optionally invert wavelengths
    if fit_config.invert_wave:
        waveffs = 1 / waveffs
        wv_to_save = 1 / wv_to_save

    # Fit with MultiFitter object
    if fit_config.multifitter == "poly":
        MultiFitter = FitMultiBandPoly(
            fitter_list=[ind_fitter_dict[f] for f in filters],
            wavelengths=waveffs,
            band_names=filters,
            linked_params=fit_config.linked_params,
            const_params=fit_config.const_params,
            wv_to_save=wv_to_save,
            **fit_config.multifitter_kwargs,
        )
    elif fit_config.multifitter == "bspline":
        MultiFitter = FitMultiBandBSpline(
            fitter_list=[ind_fitter_dict[f] for f in filters],
            wavelengths=waveffs,
            band_names=filters,
            linked_params=fit_config.linked_params,
            const_params=fit_config.const_params,
            wv_to_save=wv_to_save,
            **fit_config.multifitter_kwargs,
        )

    # Add multifitter results
    chains, summary = return_chains_summary_from_fitter(MultiFitter, band_config_dict, rkey, fit_config.method)
    sim_results_dict["chains"] = chains
    sim_results_dict["summary"] = summary
    sim_results_dict["map"] = return_MAP_from_fitter(MultiFitter, rkey)
    sim_results_dict["wv_to_save"] = wv_to_save
    sim_results_dict["waveffs"] = waveffs

    # Save results
    save_simultaneous_fit_results(
        results_dict=sim_results_dict, 
        band_config_dict=band_config_dict, 
        cube_config=cube_config, 
        model_config=model_config,
        fit_config=fit_config,
        io_config=io_config,
        )
    
    return sim_results_dict

def fit_cube(
    cube_config: CubeConfig,
    results_dict: Dict,
    fit_config: ModelConfig,
    model_config: ModelConfig,
):
    """Fit IFU cube slices using parameters from band fits"""
    
    return  # Not implemented yet

    rkey = PRNGKey(fit_config.seed)

    # Load cube data
    wave, cube, cube_err = load_cube_data(cube_config)
    nlam, ny, nx = cube.shape

    # Load values for parameters
    linked_params_dict = {
        p: return_linked_param_at_wv(results_dict, param=p, return_std=True) 
        for p in fit_config.linked_params
    }
    const_params_dict = {
        p: return_const_param(results_dict, param=p, return_std=True) 
        for p in fit_config.const_params
    }

    # Loop over each wavelength slice
    for i in range(nlam):
        slice = cube[i, :, :]
        slice_err = cube_err[i, :, :]

        # Build Gaussian priors for this slice using medians and stds
        slice_gaussian_priors = {}
        for param, (meds, stds) in linked_params_dict.items():
            if model_config.profile_type == "sersic_exp" and param != "f_1":
                slice_gaussian_priors[param] = (meds[i], stds[i])
        
        slice_const_priors = {}
        for param, (meds, stds) in const_params_dict.items():
            slice_const_priors[param] = (meds, stds)
        
        # Combine into a dict
        slice_prior_dict = {
            "uniform": {"f_1": (0.0, 1.0)},
            "gaussian": {**slice_const_priors, **slice_gaussian_priors},
        }
        
        # Create prior config for this slice
        slice_prior_config = ModelConfig(
            profile_type=model_config.profile_type,
            sky_type=model_config.sky_type,
            prior_dict=slice_prior_dict
        )
        
        # Set priors
        slice_prior = set_priors(image=slice, mask=mask, model_config=slice_prior_config)

        # Fit individual slice
        rkey, rkey_fit = jax.random.split(rkey)
        fitter, result = fit_single(
            data=slice, 
            mask=mask, 
            rms=slice_err, 
            psf=psf, 
            prior=slice_prior, 
            loss_func=fit_config.loss_func, 
            method=fit_config.method, 
            rkey=rkey_fit, 
            verbose=io_config.verbose, 
        )
        
    return fitter, result