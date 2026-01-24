"""Main Gordian class for IFU bulge-disc decomposition."""
from typing import Union

from .config import GordianConfig, load_config_from_dict
from .fitting import fit_bands_independent, fit_bands_simultaneous, fit_cube

class Gordian:
    """Base class for the Gordian package."""

    def __init__(self, config: Union[dict, GordianConfig]):
        """
        Initialize Gordian.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        print("Gordian initialised")
        
        # Convert dict to dataclass
        if isinstance(config, GordianConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config: GordianConfig = load_config_from_dict(config)
        else:
            raise Exception(f"Invalid config type: {type(config)}")

    def run_fit_bands(self):
        """Runs the fitting of photometric bands.

        Depending on the `fit_type` in the configuration, it either fits
        each band independently or performs a simultaneous fit across all bands.

        Raises
        ------
        ValueError
            If `fit_type` is not 'independent' or 'simultaneous'.
        """
        
        fit_type = self.config.fit_config.fit_type

        if fit_type == "independent":
            # Run independent fits for each band
            fit_bands_independent(
                band_config=self.config.band_config,
                model_config=self.config.model_config,
                fit_config=self.config.fit_config,
                io_config=self.config.io_config,
            )
        
        elif fit_type == "simultaneous":
            # Run a simultaneous fit across all bands and store the results
            self.bands_results_dict = fit_bands_simultaneous(
                band_config_dict=self.config.band_config,
                cube_config=self.config.cube_config,
                model_config=self.config.model_config,
                fit_config=self.config.fit_config,
                io_config=self.config.io_config,
            )
            
        else:
            raise ValueError(
                f"fit_type must be 'independent' or 'simultaneous', got {fit_type}"
            )

    def run_fit_cube(self, results_dict):
        """Runs the fitting of the IFU cube."""
        
        fit_cube(
            cube_config=self.config.cube_config,
            results_dict=results_dict,
            fit_config=self.config.fit_config,
            model_config=self.config.model_config,
            io_config=self.config.io_config,
        )

    def run(self):
        """Wrapper for running Gordian."""

        # Fit photometric bands
        self.run_fit_bands()
        
        # Fit cube if provided
        if self.config.cube_config is not None:
            self.run_fit_cube()