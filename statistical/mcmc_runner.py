import numpy as np

from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Util import util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.image_util as image_util

np.random.seed(42)


def run_mcmc_estimation(
    image,
    background_rms=0.005,
    exp_time=500.,
    deltaPix=0.05,
    fwhm=0.05,
    n_particles=500,
    n_iterations=200,
    n_burn=300,
    n_run=900
):

    numPix = image.shape[0]

    # -------------------------------------------------
    # DATA SETUP (IDENTICAL TO NOTEBOOK)
    # -------------------------------------------------

    _, _, ra_at_xy_0, dec_at_xy_0, _, _, Mpix2coord, _ = \
        util.make_grid_with_coordtransform(
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
        'image_data': image
    }

    data_class = ImageData(**kwargs_data)

    kwargs_psf = {
        'psf_type': 'GAUSSIAN',
        'fwhm': fwhm,
        'pixel_size': deltaPix,
        'truncation': 3
    }

    psf_class = PSF(**kwargs_psf)

    kwargs_numerics = {
        'supersampling_factor': 1,
        'supersampling_convolution': False
    }

    # -------------------------------------------------
    # MODEL LISTS (EXACT SAME AS NOTEBOOK)
    # -------------------------------------------------

    lens_model_list = ['SIE', 'SHEAR']
    source_model_list = ['SERSIC_ELLIPSE']
    lens_light_model_list = ['SERSIC_ELLIPSE']

    lens_model_class = LensModel(lens_model_list)
    source_model_class = LightModel(source_model_list)
    lens_light_model_class = LightModel(lens_light_model_list)

    # -------------------------------------------------
    # PARAMETER BLOCKS (EXACT STRUCTURE FROM NOTEBOOK)
    # -------------------------------------------------

    fixed_lens = []
    kwargs_lens_init = []
    kwargs_lens_sigma = []
    kwargs_lower_lens = []
    kwargs_upper_lens = []

    # SIE
    fixed_lens.append({})
    kwargs_lens_init.append({
        'theta_E': 0.7,
        'e1': 0.,
        'e2': 0.,
        'center_x': 0.,
        'center_y': 0.
    })
    kwargs_lens_sigma.append({
        'theta_E': .2,
        'e1': 0.05,
        'e2': 0.05,
        'center_x': 0.05,
        'center_y': 0.05
    })
    kwargs_lower_lens.append({
        'theta_E': 0.01,
        'e1': -0.5,
        'e2': -0.5,
        'center_x': -10,
        'center_y': -10
    })
    kwargs_upper_lens.append({
        'theta_E': 2.,
        'e1': 0.5,
        'e2': 0.5,
        'center_x': 10,
        'center_y': 10
    })

    # SHEAR
    fixed_lens.append({'ra_0': 0, 'dec_0': 0})
    kwargs_lens_init.append({'gamma1': 0., 'gamma2': 0.0})
    kwargs_lens_sigma.append({'gamma1': 0.1, 'gamma2': 0.1})
    kwargs_lower_lens.append({'gamma1': -0.2, 'gamma2': -0.2})
    kwargs_upper_lens.append({'gamma1': 0.2, 'gamma2': 0.2})

    lens_params = [
        kwargs_lens_init,
        kwargs_lens_sigma,
        fixed_lens,
        kwargs_lower_lens,
        kwargs_upper_lens
    ]

    # SOURCE
    fixed_source = [{}]
    kwargs_source_init = [{
        'R_sersic': 0.2,
        'n_sersic': 1,
        'e1': 0,
        'e2': 0,
        'center_x': 0.,
        'center_y': 0
    }]
    kwargs_source_sigma = [{
        'n_sersic': 0.5,
        'R_sersic': 0.1,
        'e1': 0.05,
        'e2': 0.05,
        'center_x': 0.2,
        'center_y': 0.2
    }]
    kwargs_lower_source = [{
        'e1': -0.5,
        'e2': -0.5,
        'R_sersic': 0.001,
        'n_sersic': .5,
        'center_x': -10,
        'center_y': -10
    }]
    kwargs_upper_source = [{
        'e1': 0.5,
        'e2': 0.5,
        'R_sersic': 10,
        'n_sersic': 5.,
        'center_x': 10,
        'center_y': 10
    }]

    source_params = [
        kwargs_source_init,
        kwargs_source_sigma,
        fixed_source,
        kwargs_lower_source,
        kwargs_upper_source
    ]

    # LENS LIGHT
    fixed_lens_light = [{}]
    kwargs_lens_light_init = [{
        'R_sersic': 0.5,
        'n_sersic': 2,
        'e1': 0,
        'e2': 0,
        'center_x': 0.,
        'center_y': 0
    }]
    kwargs_lens_light_sigma = [{
        'n_sersic': 1,
        'R_sersic': 0.3,
        'e1': 0.05,
        'e2': 0.05,
        'center_x': 0.1,
        'center_y': 0.1
    }]
    kwargs_lower_lens_light = [{
        'e1': -0.5,
        'e2': -0.5,
        'R_sersic': 0.001,
        'n_sersic': .5,
        'center_x': -10,
        'center_y': -10
    }]
    kwargs_upper_lens_light = [{
        'e1': 0.5,
        'e2': 0.5,
        'R_sersic': 10,
        'n_sersic': 5.,
        'center_x': 10,
        'center_y': 10
    }]

    lens_light_params = [
        kwargs_lens_light_init,
        kwargs_lens_light_sigma,
        fixed_lens_light,
        kwargs_lower_lens_light,
        kwargs_upper_lens_light
    ]

    kwargs_params = {
        'lens_model': lens_params,
        'source_model': source_params,
        'lens_light_model': lens_light_params
    }

    kwargs_model = {
        'lens_model_list': lens_model_list,
        'source_light_model_list': source_model_list,
        'lens_light_model_list': lens_light_model_list
    }

    kwargs_likelihood = {'source_marg': False}
    kwargs_constraints = {}

    multi_band_list = [[kwargs_data, kwargs_psf, kwargs_numerics]]

    kwargs_data_joint = {
        'multi_band_list': multi_band_list,
        'multi_band_type': 'multi-linear'
    }

    # -------------------------------------------------
    # FITTING
    # -------------------------------------------------

    fitting_seq = FittingSequence(
        kwargs_data_joint,
        kwargs_model,
        kwargs_constraints,
        kwargs_likelihood,
        kwargs_params
    )

    fitting_kwargs_list = [
        ['PSO', {
            'sigma_scale': 1.,
            'n_particles': n_particles,
            'n_iterations': n_iterations
        }],
        ['MCMC', {
            'n_burn': n_burn,
            'n_run': n_run,
            'walkerRatio': 10,
            'sigma_scale': .1
        }]
    ]

    fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()

    theta_E = kwargs_result['kwargs_lens'][0]['theta_E']

    
    from lenstronomy.Plots import chain_plot
    from lenstronomy.Plots.model_plot import ModelPlot
    import matplotlib.pyplot as plt

    modelPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string="gist_heat")

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

    modelPlot.data_plot(ax=axes[0,0])
    modelPlot.model_plot(ax=axes[0,1])
    modelPlot.normalized_residual_plot(ax=axes[0,2], v_min=-6, v_max=6)
    modelPlot.source_plot(ax=axes[1, 0], deltaPix_source=0.01, numPix=100)
    modelPlot.convergence_plot(ax=axes[1, 1], v_max=1)
    modelPlot.magnification_plot(ax=axes[1, 2])
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.show()

    f, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)

    modelPlot.decomposition_plot(ax=axes[0,0], text='Lens light', lens_light_add=True, unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1,0], text='Lens light convolved', lens_light_add=True)
    modelPlot.decomposition_plot(ax=axes[0,1], text='Source light', source_add=True, unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1,1], text='Source light convolved', source_add=True)
    modelPlot.decomposition_plot(ax=axes[0,2], text='All components', source_add=True, lens_light_add=True, unconvolved=True)
    modelPlot.decomposition_plot(ax=axes[1,2], text='All components convolved', source_add=True, lens_light_add=True, point_source_add=True)
    f.tight_layout()
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.show()
    print(kwargs_result)
    
    return {
        "theta_E": float(theta_E),
        "kwargs_result": kwargs_result
    }