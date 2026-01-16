import os
import asdf

def create_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

def save_simultaneous_fit_results(results_dict, band_config_dict, cube_config, model_config, fit_config, io_config):
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
    for key in band_config_dict.keys():
        filter = band_config_dict[key].filter
        filters.append(filter)

    # Convert dataclass configs to dicts for serialization
    band_dict = {}
    for name, band_config in band_config_dict.items():
        band_dict[name] = {
            "data_dir": band_config.data_dir,
            "data_name": band_config.data_name,
            "data_ext": band_config.data_ext,
            "centre": band_config.centre,
            "width": band_config.width,
            "height": band_config.height,
            "psf_dir": band_config.psf_dir,
            "psf_name": band_config.psf_name,
            "psf_ext": band_config.psf_ext,
            "snr_limit": band_config.snr_limit,
            "filter": band_config.filter,
        }
    
    cube_dict = None
    if cube_config is not None:
        cube_dict = {
            "data_dir": cube_config.data_dir,
            "data_name": cube_config.data_name,
            "data_ext": cube_config.data_ext,
            "wave_from_hdr": cube_config.wave_from_hdr,
            "in_wave_units": cube_config.in_wave_units,
            "out_wave_units": cube_config.out_wave_units,
            "centre": cube_config.centre,
            "width": cube_config.width,
            "height": cube_config.height,
            "wave_min": cube_config.wave_min,
            "wave_max": cube_config.wave_max,
        }
    
    model_dict = {
        "profile_type": model_config.profile_type,
        "sky_type": model_config.sky_type,
        "prior_dict": model_config.prior_dict,
    }
    
    fit_dict = {
        "fit_type": fit_config.fit_type,
        "loss_func": fit_config.loss_func,
        "method": fit_config.method,
        "multifitter": fit_config.multifitter,
        "multifitter_kwargs": fit_config.multifitter_kwargs,
        "use_cube_wave": fit_config.use_cube_wave,
        "invert_wave": fit_config.invert_wave,
        "seed": fit_config.seed,
        "linked_params": fit_config.linked_params,
        "const_params": fit_config.const_params,
    }

    # Create data tree
    tree = {
        "filters": filters,
        "results_dict": results_dict,
        "model_config": model_dict,
        "band_config": band_dict,
        "cube_config": cube_dict,
        "fit_config": fit_dict,
    }

    # Build output filename
    profile_str = model_config.profile_type
    method_str = fit_config.method
    multifit_str = fit_config.multifitter
    if multifit_str == "poly":
        poly_order = fit_config.multifitter_kwargs.get("poly_order", "")
        multifit_str = f"{multifit_str}{poly_order}"
    elif multifit_str == "bspline":
        n_knots = fit_config.multifitter_kwargs.get("N_knots", "")
        multifit_str = f"{multifit_str}{n_knots}"
    out_file = f"{profile_str}_{method_str}_simfit_{multifit_str}_results.asdf"

    print(f"Saved simultaneous fit result to: {out_file}")

    # Save results
    af = asdf.AsdfFile(tree)
    out_path = os.path.join(io_config.out_dir, out_file)
    af.write_to(out_path)

def save_individual_fit_result(result, model_config, fit_config, io_config):

    # Build output filename
    out_file = f"{model_config.profile_type}_{fit_config.method}_fit_{filter}.asdf"
    out_path = os.path.join(io_config.out_dir, out_file)

    # Save result
    result.save_result(out_path)