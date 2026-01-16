import os

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pysersic.results import plot_residual, plot_image
from pysersic.rendering import HybridRenderer

def make_plots(fitter, image, mask, sig, psf, profile_type, method, filter, fig_dir):

    # Plot input data 
    fig = call_plot_image(image, mask, sig, psf)
    fig_name = f"{profile_type}_{method}_{filter}_data.pdf"
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close(fig)

    # Plot model comparison
    fig = call_plot_residual(fitter, image, mask, psf, profile_type)
    fig_name = f"{profile_type}_{method}_{filter}_residual.pdf"
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close(fig)
    
    # Plot posteriors
    fig = fitter.sampling_results.corner(color='C0')
    fig_name = f"{profile_type}_{method}_{filter}_corner.pdf"
    fig.savefig(os.path.join(fig_dir, fig_name))
    plt.close(fig)

def call_plot_residual(fitter, im, mask, psf, profile_type):

    # Generate best model
    summary = fitter.sampling_results.summary()
    dict = {}
    for hdul, b in zip(summary.index, summary["mean"]):
        dict[hdul] = b
    best_model = HybridRenderer(im.shape, jnp.array(psf.astype(np.float32))).render_source(dict, profile_type=profile_type)
    
    # Call plot_residual
    fig, ax = plot_residual(im, best_model, mask=mask, vmin=-1, vmax=1)
    fig.tight_layout()

    return fig

def call_plot_image(image, mask, sig, psf):

    # Call plot_image
    fig, ax = plot_image(image, mask, sig, psf)
    fig.tight_layout()

    return fig

def plot_data_grid(trees, filters):

    # Create figure
    ncols = 3
    nrows = len(trees)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4*nrows), squeeze=False)

    # Loop over each tree
    for i, (tree, filter) in enumerate(zip(trees, filters)):
        ax = axes[i, :]

        # Load data
        input_data = tree["input_data"]
        image = np.asarray(input_data["image"])
        mask = np.asarray(input_data["mask"])
        psf_np = np.asarray(input_data["psf"]).astype(np.float32)

def plot_residuals_grid(images, masks, models, residuals, filters : str, scale : float = 2.0, resid_vmin : float = -1, resid_vmax : float = 1):

    # Create figure
    ncols = 3
    nrows = len(images)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4*nrows), squeeze=False)

    # TODO: Apply mask...
    # for i, (image, mask, model, residual, filters) in enumerate(zip(images, masks, models, residuals, filters)):
    for i, (image, model, residual, mask, filter) in enumerate(zip(images, models, residuals, masks, filters)):
        ax = axes[i, :]

        # Apply mask
        image_masked = image.copy()
        model_masked = model.copy()
        image_masked[mask.astype(bool)] = np.nan
        model_masked[mask.astype(bool)] = np.nan

        # Scale plot limits
        vmin = np.nanmean(image) - (scale * np.nanstd(image, ddof=1))
        vmax = np.nanmean(image) + (scale * np.nanstd(image, ddof=1))
        
        # Plot data
        ax[0].imshow(image_masked, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        ax[0].text(0.03, 0.97, f"{filter.upper()}", transform=ax[0].transAxes, va="top", ha="left", size=14)

        # Plot model
        ax[1].imshow(model_masked, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
        
        # Plot residual
        im_resid = ax[2].imshow(residual, origin="lower", cmap="seismic", vmin=resid_vmin, vmax=resid_vmax)
        ax_divider = make_axes_locatable(ax[2])
        cax = ax_divider.append_axes("right", size="7%", pad="2%")
        cb = fig.colorbar(im_resid, cax=cax)

    fig.tight_layout()

    return fig