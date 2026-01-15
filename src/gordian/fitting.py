import os
from typing import Dict, Optional

import numpy as np
from sedpy.observate import load_filters

import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from pysersic import FitSingle
from pysersic.multiband import FitMultiBandPoly, FitMultiBandBSpline
from pysersic.priors import autoprior
from pysersic import loss

from .config import BandConfig, CubeConfig, PriorConfig, FitConfig, IOConfig
from .loading import load_band_data, load_cube_data
from .plotting import call_plot_residual, call_plot_image, make_plots
from .helpers import (
    return_linked_param_at_wv, 
    return_const_param, 
    return_MAP_from_fitter, 
    return_summary_from_estimation
)


def _set_priors(image, mask, prior_config: PriorConfig):
    """Sets priors for model in PySersic"""
    
    # Generate priors from image
    prior = autoprior(
        image=image, 
        profile_type=prior_config.profile_type, 
        mask=mask, 
        sky_type=prior_config.sky_type
    )

    # Set priors from config dict
    for prior_type, prior_type_dict in prior_config.prior_dict.items():
        # -- uniform priors
        if prior_type == "uniform":
            for param, range in prior_type_dict.items():
                lo, hi = range
                prior.set_uniform_prior(param, lo, hi)
        # -- gaussian priors
        if prior_type == "gaussian":
            for param, gauss in prior_type_dict.items():
                loc, std = gauss
                prior.set_gaussian_prior(param, loc, std)

    return prior


def fit_single(data, mask, rms, psf, prior, loss_func, method, rkey, verbose):
    """Fits single image. Either photometric band or slice/set of slices from IFU cube"""

    if verbose:
        print("Starting fit...")

    loss_map = {
        "student_t_loss": loss.student_t_loss,
        "student_t_loss_free_sys": loss.student_t_loss_free_sys,
        "gaussian_loss": loss.gaussian_loss,
        "cash_loss": loss.cash_loss,
    }
    loss_func = loss_map[loss_func]

    # Setup fitter
    fitter = FitSingle(data=data, rms=rms, mask=mask, psf=psf, prior=prior, loss_func=loss_func)

    # Estimate posterior
    rkey, rkey_est = jax.random.split(rkey, 2)
    # -- MCMC sampling
    if method == "mcmc":
        fitter.sample(rkey=rkey_est)
        result = fitter.sampling_results
    # -- SVI
    elif method == "svi":
        result = fitter.estimate_posterior(rkey=rkey_est)
    
    if verbose:
        print("Result from fit:")
        if method == "mcmc":
            print(result.retrieve_param_quantiles(return_dataframe=True))

    return fitter, result


def fit_bands_independent(
    band_config: Dict[str, BandConfig],
    prior_config: PriorConfig,
    fit_config: FitConfig,
    io_config: IOConfig,
):
    """Fit bands independently"""
    
    rkey = PRNGKey(fit_config.seed)

    # Loop over each band
    for band_name, band_config in band_config.items():

        filter = band_config.filter
        sedpy_filter = load_filters(["jwst_" + filter])[0]
        if fit_config.verbose:
            print("------------------------------")
            print(f"Filter: {filter.upper()}")

        # Load cutout data
        image, mask, sig, psf = load_band_data(band_config)

        # Set priors
        prior = _set_priors(image=image, mask=mask, prior_config=prior_config)

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
            verbose=fit_config.verbose
        )

        # Save results
        out_name = f"{prior_config.profile_type}_{fit_config.method}_fit_{filter}.asdf"
        out_path = os.path.join(io_config.out_dir, out_name)
        result.save_result(out_path)

        # Make plots
        make_plots(
            fitter, image, mask, sig, psf, 
            profile_type=prior_config.profile_type, 
            method=fit_config.method, 
            filter=filter, 
            fig_dir=io_config.fig_dir
        )

    return fitter, result


def fit_bands_simultaneous(
    band_config: Dict[str, BandConfig],
    cube_config: Optional[CubeConfig],
    prior_config: PriorConfig,
    fit_config: FitConfig,
    io_config: IOConfig,
):
    """Fit bands simultaneously with wavelength-dependent parameters"""
    
    rkey = PRNGKey(fit_config.seed)

    filters = []
    waveffs = []

    im_dict = {}
    mask_dict = {}
    rms_dict = {}
    psf_dict = {}
    pysersic_prior_dict = {}

    ind_fitter_dict = {}
    ind_results_dict = {}
    sim_results_dict = {}

    # Loop over each band
    for band_name, band_config in band_config.items():

        filter = band_config.filter
        sedpy_filter = load_filters(["jwst_" + filter])[0]
        if fit_config.verbose:
            print("------------------------------")
            print(f"Filter: {filter.upper()}")

        # Load cutout data
        image, mask, sig, psf = load_band_data(band_config)

        # Create PySersic priors
        prior = _set_priors(image=image, mask=mask, prior_config=prior_config)
        if fit_config.verbose:
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
            verbose=fit_config.verbose,
        )

        # Save results
        out_name = f"{prior_config.profile_type}_{fit_config.method}_fit_{filter}.asdf"
        out_path = os.path.join(io_config.out_dir, out_name)
        result.save_result(out_path)

        # Make plots
        make_plots(
            fitter, image, mask, sig, psf, 
            profile_type=prior_config.profile_type, 
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
        pysersic_prior_dict[filter] = prior
        ind_fitter_dict[filter] = fitter
        ind_results_dict[filter] = result

    # Create fine grid of wavelengths
    waveffs = np.asarray(waveffs)
    if fit_config.use_cube_wave and cube_config is not None:
        wave, _, _, _ = load_cube_data(cube_config)
        wv_to_save = wave
    else:
        wv_to_save = np.linspace(min(waveffs), max(waveffs), num=50)
    
    # Invert wavelengths if requested
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
    
    # Add multifit results
    sim_results_dict["summary"] = return_summary_from_estimation(MultiFitter, rkey, fit_config.method)
    sim_results_dict["map"] = return_MAP_from_fitter(MultiFitter, rkey)
    sim_results_dict["wv_to_save"] = wv_to_save
    sim_results_dict["waveffs"] = waveffs

    return sim_results_dict


def fit_cube(
    cube_config: CubeConfig,
    results_dict: Dict,
    fit_config: FitConfig,
    prior_config: PriorConfig,
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
            if prior_config.profile_type == "sersic_exp" and param != "f_1":
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
        slice_prior_config = PriorConfig(
            profile_type=prior_config.profile_type,
            sky_type=prior_config.sky_type,
            prior_dict=slice_prior_dict
        )
        
        # Set priors
        slice_prior = _set_priors(image=slice, mask=mask, prior_config=slice_prior_config)

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
            verbose=fit_config.verbose, 
        )
        
    return fitter, result