"""Configuration dataclasses for Gordian pipeline."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class BandConfig:
    """Configuration for a single photometric band."""
    data_dir: str
    data_name: str
    data_ext: str
    centre: Tuple[int, int]
    width: int
    height: int
    psf_dir: str
    psf_name: str
    psf_ext: Optional[str]
    snr_limit: float
    filter: str


@dataclass
class CubeConfig:
    """Configuration for IFU cube data."""
    data_dir: str
    data_name: str
    data_ext: str
    wave_from_hdr: bool
    in_wave_units: Optional[str]
    out_wave_units: Optional[str]
    centre: Tuple[int, int]
    width: int
    height: int
    wave_min: Optional[float]
    wave_max: Optional[float]
    psf_dir: Optional[str] = None
    psf_name: Optional[str] = None
    psf_ext: Optional[str] = None


@dataclass
class ModelConfig:
    """Prior configuration for model parameters."""
    profile_type: str
    sky_type: str
    prior_dict: Dict


@dataclass
class FitConfig:
    """Fitting configuration."""
    fit_type: str
    loss_func: str
    method: str
    multifitter: str
    multifitter_kwargs: Dict
    use_cube_wave: bool
    invert_wave: bool
    seed: int
    linked_params: List[str]
    const_params: List[str]


@dataclass
class IOConfig:
    """I/O configuration."""
    out_dir: str
    fig_dir: str
    verbose: bool


@dataclass
class GordianConfig:
    """Complete Goridan pipeline configuration."""
    io_config: IOConfig
    band_config: Dict[str, BandConfig]
    cube_config: Optional[CubeConfig]
    model_config: ModelConfig
    fit_config: ModelConfig


def load_config_from_dict(config_dict: dict) -> GordianConfig:
    """
    Convert a configuration dictionary to GordianConfig dataclass.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary
        
    Returns
    -------
    GordianConfig
        Validated configuration object
    """
    
    # Build I/O config
    io = IOConfig(**config_dict["io"])
    
    # Build band configs
    band_config = {}
    for band_name, band_data in config_dict["band_config"].items():
        band_config[band_name] = BandConfig(**band_data)
    
    # Build cube config if present
    cube_config = None
    if "cube_config" in config_dict and config_dict["cube_config"] is not None:
        cube_config = CubeConfig(**config_dict["cube_config"])
    
    # Build prior config
    model_config = ModelConfig(**config_dict["model_config"])
    
    # Build fit config
    fit_config = ModelConfig(**config_dict["fit_config"])
    
    return GordianConfig(
        io_config=io,
        band_config=band_config,
        cube_config=cube_config,
        model_config=model_config,
        fit_config=fit_config
    )