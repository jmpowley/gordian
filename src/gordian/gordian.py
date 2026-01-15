"""Main Gordian class for IFU bulge-disc decomposition."""

from .config import GordianConfig, load_config_from_dict
from .fitting import fit_bands_independent, fit_bands_simultaneous, fit_cube
from .writing import save_simultaneous_fit_results_to_asdf, create_directory


class Gordian:
    """Performs bulge-disc decomposition of IFU cubes"""

    def __init__(self, config: dict):
        """
        Initialize Gordian pipeline.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        print("Gordian initialised")
        
        # Convert dict to dataclass
        if isinstance(config, dict):
            self.config: GordianConfig = load_config_from_dict(config)
        else:
            self.config = config

    def fit_bands(self):
        """Fit photometric bands to obtain structural parameters"""
        
        fit_type = self.config.fit_config.fit_type
        
        if fit_type == "independent":
            return fit_bands_independent(
                band_config=self.config.band_config,
                prior_config=self.config.prior_config,
                fit_config=self.config.fit_config,
                io_config=self.config.io_config,
            )
        elif fit_type == "simultaneous":
            return fit_bands_simultaneous(
                band_config=self.config.band_config,
                cube_config=self.config.cube_config,
                prior_config=self.config.prior_config,
                fit_config=self.config.fit_config,
                io_config=self.config.io_config,
            )
        else:
            raise ValueError(f"fit_type must be 'independent' or 'simultaneous', got {fit_type}")

    def fit_cube(self, results_dict):
        """Fit cube slices using structural parameters from fit"""
        
        if self.config.cube_config is None:
            print("No cube configuration provided. Skipping cube fitting.")
            return
        
        fit_cube(
            cube_config=self.config.cube_config,
            results_dict=results_dict,
            fit_config=self.config.fit_config,
            prior_config=self.config.prior_config,
        )

    def run(self):
        """Wrapper for Gordian. Prepares data for fitting and then runs pipeline"""
        
        # Initialise directories
        create_directory(self.config.io_config.fig_dir)
        create_directory(self.config.io_config.out_dir)

        # Fit photometric bands to obtain structural parameters
        bands_results_dict = self.fit_bands()
        
        # Save results for simultaneous fits
        if self.config.fit_config.fit_type == "simultaneous":
            save_simultaneous_fit_results_to_asdf(
                results_dict=bands_results_dict,
                config=self.config,
            )
        
        # Extract structural components from cube if provided
        if self.config.cube_config is not None:
            self.fit_cube(results_dict=bands_results_dict)