import numpy as np
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.image_util as image_util


def simulate_lens_case(
    kwargs_lens,
    kwargs_source,
    kwargs_lens_light,
    background_rms=0.005,
    exp_time=500.0,
    numPix=128,
    deltaPix=0.05,
    fwhm=0.05,
    psf_type='GAUSSIAN'
):

    # --- Coordinate grid ---
    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = util.make_grid_with_coordtransform(
        numPix=numPix,
        deltapix=deltaPix,
        center_ra=0,
        center_dec=0,
        subgrid_res=1,
        inverse=False
    )

    kwargs_data = {
        'background_rms': background_rms,
        'exposure_time': exp_time,
        'ra_at_xy_0': ra_at_xy_0,
        'dec_at_xy_0': dec_at_xy_0,
        'transform_pix2angle': Mpix2coord,
        'image_data': np.zeros((numPix, numPix))
    }

    data_class = ImageData(**kwargs_data)

    # --- PSF ---
    kwargs_psf = {
        'psf_type': psf_type,
        'fwhm': fwhm,
        'pixel_size': deltaPix,
        'truncation': 3
    }
    psf_class = PSF(**kwargs_psf)

    # --- Models ---
    lens_model_class = LensModel(['SIE', 'SHEAR'])
    source_model_class = LightModel(['SERSIC_ELLIPSE'])
    lens_light_model_class = LightModel(['SERSIC_ELLIPSE'])

    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }

    imageModel = ImageModel(
        data_class,
        psf_class,
        lens_model_class=lens_model_class,
        source_model_class=source_model_class,
        lens_light_model_class=lens_light_model_class,
        kwargs_numerics=kwargs_numerics
    )

    # --- Full forward model (lens + source + lens light) ---
    image_model = imageModel.image(
        kwargs_lens,
        kwargs_source,
        kwargs_lens_light=kwargs_lens_light,
        kwargs_ps=None
    )

    # --- Lensed source only (NO lens light) ---
    source_image = imageModel.source_surface_brightness(
        kwargs_source,
        kwargs_lens=kwargs_lens
    )

    # --- Add noise ---
    poisson = image_util.add_poisson(image_model, exp_time=exp_time)
    bkg = image_util.add_background(image_model, sigma_bkd=background_rms)
    image_real = image_model + poisson + bkg

    return {
        "observed": image_real,
        "model": image_model,
        "source": source_image
    }
