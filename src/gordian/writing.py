import os
import asdf

from .config import GordianConfig

def create_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def save_simultaneous_fit_results_to_asdf(results_dict, config: GordianConfig):
    """
    Write simultaneous fit results and configuration to an ASDF file.
    
    Parameters
    ----------
    results_dict : dict
        Results from simultaneous fitting
    config : GordianConfig
        Complete pipeline configuration
    """

    # Extract filters
    filters = []
    for key in config.band_config.keys():
        filter = config.band_config[key].filter
        filters.append(filter)

    # Convert dataclass configs to dicts for serialization
    band_dict = {}
    for name, band in config.band_config.items():
        band_dict[name] = {
            "data_dir": band.data_dir,
            "data_name": band.data_name,
            "data_ext": band.data_ext,
            "centre": band.centre,
            "width": band.width,
            "height": band.height,
            "psf_dir": band.psf_dir,
            "psf_name": band.psf_name,
            "psf_ext": band.psf_ext,
            "snr_limit": band.snr_limit,
            "filter": band.filter,
        }
    
    cube_dict = None
    if config.cube_config is not None:
        cube_dict = {
            "data_dir": config.cube_config.data_dir,
            "data_name": config.cube_config.data_name,
            "data_ext": config.cube_config.data_ext,
            "wave_from_hdr": config.cube_config.wave_from_hdr,
            "in_wave_units": config.cube_config.in_wave_units,
            "out_wave_units": config.cube_config.out_wave_units,
            "centre": config.cube_config.centre,
            "width": config.cube_config.width,
            "height": config.cube_config.height,
            "wave_min": config.cube_config.wave_min,
            "wave_max": config.cube_config.wave_max,
        }
    
    prior_dict = {
        "profile_type": config.prior_config.profile_type,
        "sky_type": config.prior_config.sky_type,
        "prior_dict": config.prior_config.prior_dict,
    }
    
    fit_dict = {
        "fit_type": config.fit_config.fit_type,
        "loss_func": config.fit_config.loss_func,
        "method": config.fit_config.method,
        "multifitter": config.fit_config.multifitter,
        "multifitter_kwargs": config.fit_config.multifitter_kwargs,
        "use_cube_wave": config.fit_config.use_cube_wave,
        "invert_wave": config.fit_config.invert_wave,
        "seed": config.fit_config.seed,
        "linked_params": config.fit_config.linked_params,
        "const_params": config.fit_config.const_params,
    }

    # Create data tree
    tree = {
        "results_dict": results_dict,
        "prior_config": prior_dict,
        "filters": filters,
        "band_config": band_dict,
        "cube_config": cube_dict,
        "fit_config": fit_dict,
    }

    # Build output filename
    profile_str = config.prior_config.profile_type
    method_str = config.fit_config.method
    multifit_str = config.fit_config.multifitter
    
    if multifit_str == "poly":
        poly_order = config.fit_config.multifitter_kwargs.get("poly_order", "")
        multifit_str = f"{multifit_str}{poly_order}"
    elif multifit_str == "bspline":
        n_knots = config.fit_config.multifitter_kwargs.get("N_knots", "")
        multifit_str = f"{multifit_str}{n_knots}"
    
    out_name = f"{profile_str}_{method_str}_simfit_{multifit_str}_results.asdf"

    print(f"Saved simultaneous fit result to: {out_name}")

    # Save results
    af = asdf.AsdfFile(tree)
    out_path = os.path.join(config.io_config.out_dir, out_name)
    af.write_to(out_path)