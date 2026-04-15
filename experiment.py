from astropy.io import fits
from pipeline.run_pipeline import full_pipeline
from inference.likelihood import log_probability

with fits.open("jwst_image.fits") as hdul:
    image = hdul[1].data.astype(float)

results = full_pipeline(
    image=image,
    pixel_scale=0.031,  # JWST NIRCam
    log_probability=log_probability
)
