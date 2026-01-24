import os
from typing import Optional, Tuple, Literal

import numpy as np
from astropy.io import fits

from .config import BandConfig, CubeConfig


def resolve_path(*, file_path: Optional[str] = None, file_dir: Optional[str] = None, file_name: Optional[str] = None):
    """
    Resolves a file path. Either verify an existing file path, or construct a path from a directory and file name.
    """

    # If file_path supplied, check it is correct
    if file_path is not None:
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"The following file path does not exist: {file_path}")
        
    # If file_dir and file_name supplied, check combination is corret
    elif file_dir is not None and file_name is not None:
        file_path = os.path.join(file_dir, file_name)
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"The following file path does not exist: {file_path}")
    else:
        raise Exception("Either file_path or file_dir and file_name must be provided")


def rebuild_wave_from_header(header):

    try:
        crval = header["CRVAL3"]
        cdelt = header["CDELT3"]
        naxis = header["NAXIS3"]

        pix = np.arange(naxis)
        wave = crval + pix * cdelt

    except Exception as e:
        return e
    
    return wave


def extract_subregion(data, centre: Tuple[int, int], height: int, width: int, n_pad: int = 0) -> np.ndarray:

    # Check padding size
    if 2 * n_pad >= height or 2 * n_pad >= width:
        raise ValueError("n_pad too large: padded region has non-positive size")

    # Extract co-ordinates
    ycen, xcen = centre
    halfw = width // 2
    halfh = height // 2

    # Create region (with optional padding)
    y0 = ycen - halfh + n_pad
    y1 = y0 + (height - 2*n_pad)
    x0 = xcen - halfw + n_pad
    x1 = x0 + (width - 2*n_pad)

    # Check subregion within bounds of data
    # -- image
    if data.ndim == 2:
        ny, nx = data.shape
        if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
            raise IndexError("Requested subregion extends outside data array")
        return data[y0:y1, x0:x1]
    # -- cube
    elif data.ndim == 3:
        ny, nx = data.shape[1], data.shape[2]
        if y0 < 0 or x0 < 0 or y1 > ny or x1 > nx:
            raise IndexError("Requested subregion extends outside data array")
        return data[:, y0:y1, x0:x1]
    else:
        raise ValueError("Input data must be 2D or 3D")

def normalise(data):

    if data.ndim == 2:
        data /= np.nansum(data)
    elif data.ndim == 3:
        nlam, ny, nx = data.shape
        for i in range(nlam):
            data[i, :, :] = data[i, :, :] / np.nansum(data[i, :, :])

    return data.astype(float)


def load_band_data(band_config: BandConfig):
    """
    Load cutout data for a single band.
    
    Parameters
    ----------
    band_config : BandConfig
        Band configuration object
        
    Returns
    -------
    image : ndarray
        Image data
    mask : ndarray
        Mask array
    sig : ndarray
        Uncertainty array
    psf : ndarray
        PSF array
    """

    # Load image data
    data_path = resolve_path(file_dir=band_config.data_dir, file_name=band_config.data_name)
    image = fits.getdata(data_path, extname=band_config.data_ext)
    wht = fits.getdata(data_path, extname="WHT")
    
    # Create mask
    mask = np.zeros_like(image)

    # Impose SNR limit
    sig = (1 / band_config.snr_limit) * np.sqrt(np.abs(image))
    sig[sig==0] = (1 / band_config.snr_limit) * np.sqrt(np.mean(np.abs((image))))

    # Only extract subregion if centre, width, height provided
    if band_config.centre is not None and band_config.width is not None and band_config.height is not None:
        image = extract_subregion(image, band_config.centre, band_config.height, band_config.width)
        sig = extract_subregion(sig, band_config.centre, band_config.height, band_config.width)
        wht = extract_subregion(wht, band_config.centre, band_config.height, band_config.width)
        mask = extract_subregion(mask, band_config.centre, band_config.height, band_config.width)

    # Load PSF data
    psf = None
    if band_config.psf_dir is not None and band_config.psf_name is not None:
        psf_path = resolve_path(file_dir=band_config.psf_dir, file_name=band_config.psf_name)
        psf = fits.getdata(psf_path, extname=band_config.psf_ext if band_config.psf_ext is not None else "PRIMARY")
        
        # extract subregion
        if band_config.width is not None and band_config.height is not None:
            psf_ny, psf_nx = psf.shape
            psf_centre = (int(psf_ny // 2), int(psf_nx // 2))
            psf = extract_subregion(psf, psf_centre, band_config.height, band_config.width, n_pad=2)
        
        # normalize along each spectral axis
        psf = normalise(psf)

    return image, mask, sig, psf


def load_cube_data(cube_config: CubeConfig):
    """
    Load IFU cube data.
    
    Parameters
    ----------
    cube_config : CubeConfig
        Cube configuration object
        
    Returns
    -------
    wave : ndarray
        Wavelength array
    cube : ndarray
        Cube data
    err : ndarray
        Error cube
    """

    # Load cube data
    data_path = resolve_path(file_dir=cube_config.data_dir, file_name=cube_config.data_name)
    if cube_config.wave_from_hdr:
        _, wave_hdr = fits.getdata(data_path, extname=cube_config.data_ext, header=True)
        wave = rebuild_wave_from_header(wave_hdr)
    cube = fits.getdata(data_path, extname=cube_config.data_ext)
    err = fits.getdata(data_path, extname="ERR")

    # Convert wavelength units
    # TODO: Apply conversions using cube_config.in_wave_units and cube_config.out_wave_units

    # Extract subregion
    if cube_config.centre is not None and cube_config.width is not None and cube_config.height is not None:
        cube = extract_subregion(cube, cube_config.centre, cube_config.height, cube_config.width)
        err = extract_subregion(err, cube_config.centre, cube_config.height, cube_config.width)

    # Load PSF data if provided
    psf = None
    if cube_config.psf_dir is not None and cube_config.psf_name is not None:
        psf_path = resolve_path(file_dir=cube_config.psf_dir, file_name=cube_config.psf_name)
        psf = fits.getdata(psf_path, extname=cube_config.psf_ext if cube_config.psf_ext is not None else "PRIMARY")
        
        # extract subregion
        if cube_config.height is not None and cube_config.width is not None:
            psf_ny, psf_nx = psf.shape
            psf_centre = int(0.5*psf_ny), int(0.5*psf_nx)
            psf = extract_subregion(psf, psf_centre, cube_config.height, cube_config.width, n_pad=2)
        
        # normalise along each spectral axis
        psf = normalise(psf)

    # Crop wavelength axis
    if cube_config.wave_min is not None and cube_config.wave_max is not None:
        wave_mask = (wave > cube_config.wave_min) & (wave < cube_config.wave_max)
        wave = wave[wave_mask]
        cube = cube[wave_mask, :, :]
        err = err[wave_mask, :, :]
        if cube_config.psf_dir is not None:
            psf = psf[wave_mask, :, :]
    
    return wave, cube, err, psf