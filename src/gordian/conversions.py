import numpy as np
import astropy.units as u

# --------------------------
# Define constants for units
# --------------------------
MAGGIE_UNIT = u.def_unit("maggie", 3631 * u.Jy)
CGS_FLUX_UNIT = u.erg / (u.s * u.cm**2 * u.AA)
SI_FLUX_UNIT = u.W / u.m**3

# -------------------
# Conversion functions
# -------------------
def convert_wave(wave_in, in_unit, out_unit, return_quantity=False):
    """Convert wavelength between units
    
    Parameters
    ----------
    wave_in : float or array
        Input wavelength values
    in_unit : Unit
        Input wavelength unit
    out_unit : Unit
        Output wavelength unit
    return_quantity : bool
        Return as astropy Quantity (default: False)
    
    Returns
    -------
    wave_out : float, array, or Quantity
        Converted wavelength
    """
    
    # Assign units
    if not isinstance(wave_in, u.Quantity):
        wave_in = wave_in * in_unit
    
    # Apply conversion
    try:
        wave_out = wave_in.to(out_unit)
    except Exception as e:
        raise ValueError(f"Error converting wavelength from {in_unit} to {out_unit}: {e}")

    # Return units
    if return_quantity:
        return wave_out
    else:
        return wave_out.value


def convert_flux(flux_in, in_unit, out_unit, err_in=None, wave=None, return_quantity=False):
    """Convert flux and optional error between different units
    
    Handles special cases:
    - Spectral density conversions (requires wavelength)
    - Magnitude conversions (logarithmic transformation)
    - Maggie conversions (dimensionless flux unit)
    
    Parameters
    ----------
    flux_in : float or array or Quantity
        Input flux values
    in_unit : Unit
        Input flux unit
    out_unit : Unit
        Output flux unit
    err_in : float or array or Quantity, optional
        Input flux error/uncertainty
    wave : Quantity, optional
        Wavelength (required for spectral density conversions)
    return_quantity : bool
        Return as astropy Quantity (default: False)
    
    Returns
    -------
    flux_out : float, array, or Quantity
        Converted flux values
    err_out : float, array, or Quantity or None
        Converted flux error (None if err_in was None)
    """
    
    # Assign units to flux
    if not isinstance(flux_in, u.Quantity):
        flux_in = flux_in * in_unit
    
    # Assign units to error
    if err_in is not None and not isinstance(err_in, u.Quantity):
        err_in = err_in * in_unit
    
    # Converting from magnitudes
    if in_unit == u.mag:
        flux_out, err_out = convert_flux_from_magnitudes(flux_in, out_unit, err_in=err_in, wave=wave)

    # Converting to magnitudes
    elif out_unit == u.mag:
        flux_out, err_out = convert_flux_to_magnitudes(flux_in, in_unit, err_in=err_in, wave=wave)
    
    # Standard conversion
    else:
        flux_out, err_out = convert_flux_standard(flux_in, in_unit, out_unit, err_in=err_in, wave=wave)
    
    # Return results (always return both flux and error)
    if return_quantity:
        return flux_out, err_out
    else:
        return flux_out.value, (None if err_out is None else err_out.value)


def convert_flux_from_magnitudes(flux_in, out_unit, err_in=None, wave=None):
    """Convert from magnitude to other flux units
    
    Parameters
    ----------
    flux_in : Quantity
        Flux in magnitudes
    out_unit : Unit
        Target flux unit
    err_in : Quantity, optional
        Flux error in magnitudes
    wave : Quantity, optional
        Wavelength (required for spectral density conversions)
    
    Returns
    -------
    flux_out : Quantity
        Converted flux
    err_out : Quantity or None
        Converted error (None if err_in was None)
    """
    
    # Convert Magnitude to maggie
    flux_maggie = 10.0 ** (-0.4 * flux_in.value)
    if err_in is not None:
        # Error propagation: dF/F ≈ 0.921 * dm
        factor = 0.4 * np.log(10.0)
        err_maggie = np.abs(flux_maggie * factor * err_in.value)
    else:
        err_maggie = None
    
    # Convert maggie to target unit
    if out_unit == u.mag:
        # mag → mag (no conversion)
        flux_out = flux_in
        err_out = err_in
    else:
        flux_maggie_qty = flux_maggie * u.dimensionless_unscaled
        if err_maggie is not None:
            err_maggie_qty = err_maggie * u.dimensionless_unscaled
        else:
            err_maggie_qty = None
        
        # Convert to Jy first
        flux_jy = flux_maggie_qty * 3631 * u.Jy
        if err_maggie_qty is not None:
            err_jy = err_maggie_qty * 3631 * u.Jy
        else:
            err_jy = None
        
        # Convert Jy to target unit
        if out_unit == u.Jy:
            flux_out = flux_jy
            err_out = err_jy
        elif out_unit == MAGGIE_UNIT:
            flux_out = flux_jy / 3631
            err_out = err_jy / 3631 if err_jy is not None else None
        else:
            # Spectral density conversion (requires wavelength)
            if wave is None:
                raise ValueError(f"Wavelength required for conversion from magnitude to {out_unit}")
            flux_out = flux_jy.to(out_unit, equivalencies=u.spectral_density(wave))
            if err_jy is not None:
                err_out = err_jy.to(out_unit, equivalencies=u.spectral_density(wave))
            else:
                err_out = None

    return flux_out, err_out


def convert_flux_to_magnitudes(flux_in, in_unit, err_in=None, wave=None):
    """Convert from other flux units to magnitude
    
    Parameters
    ----------
    flux_in : Quantity
        Input flux
    in_unit : Unit
        Input flux unit
    err_in : Quantity, optional
        Flux error
    wave : Quantity, optional
        Wavelength (required for spectral density conversions)
    
    Returns
    -------
    flux_out : Quantity
        Flux in magnitudes
    err_out : Quantity or None
        Error in magnitudes (None if err_in was None)
    """

    # Convert to Jy first
    if in_unit == u.Jy:
        flux_jy = flux_in
        err_jy = err_in
    elif in_unit == MAGGIE_UNIT:
        flux_jy = flux_in * 3631 * u.Jy
        err_jy = err_in * 3631 * u.Jy if err_in is not None else None
    else:
        # Spectral density conversion (requires wavelength)
        if wave is None:
            raise ValueError(f"Wavelength required for conversion from {in_unit} to magnitude")
        flux_jy = flux_in.to(u.Jy, equivalencies=u.spectral_density(wave))
        if err_in is not None:
            err_jy = err_in.to(u.Jy, equivalencies=u.spectral_density(wave))
        else:
            err_jy = None
    
    # Convert Jy to maggie
    flux_maggie = flux_jy / 3631
    if err_jy is not None:
        err_maggie = err_jy / 3631
    else:
        err_maggie = None
    
    # Convert maggie to magnitude
    flux_mag = -2.5 * np.log10(flux_maggie.value)
    if err_maggie is not None:
        # Error propagation: dm ≈ 1.086 * dF/F
        err_mag = 2.5 / np.log(10.0) * err_maggie.value / flux_maggie.value
    else:
        err_mag = None
    
    flux_out = flux_mag * u.mag
    err_out = err_mag * u.mag if err_mag is not None else None

    return flux_out, err_out


def convert_flux_standard(flux_in, in_unit, out_unit, err_in=None, wave=None):
    """Standard flux conversion (no magnitudes involved)
    
    Parameters
    ----------
    flux_in : Quantity
        Input flux
    in_unit : Unit
        Input flux unit
    out_unit : Unit
        Output flux unit
    err_in : Quantity, optional
        Flux error
    wave : Quantity, optional
        Wavelength (required for spectral density conversions)
    
    Returns
    -------
    flux_out : Quantity
        Converted flux
    err_out : Quantity or None
        Converted error (None if err_in was None)
    """

    try:
        # Try simple conversion first
        flux_out = flux_in.to(out_unit)
        if err_in is not None:
            err_out = err_in.to(out_unit)
        else:
            err_out = None
    except Exception:
        # Try spectral density conversion
        if wave is None:
            raise ValueError(
                f"Conversion from {in_unit} to {out_unit} failed. "
                f"If this is a spectral-density conversion, you must supply wavelength."
            )
        flux_out = flux_in.to(out_unit, equivalencies=u.spectral_density(wave))
        if err_in is not None:
            err_out = err_in.to(out_unit, equivalencies=u.spectral_density(wave))
        else:
            err_out = None

    return flux_out, err_out