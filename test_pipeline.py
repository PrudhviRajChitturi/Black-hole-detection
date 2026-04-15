import numpy as np
from astropy.io import fits
from run_pipeline import run_pipeline, visualize_results

with fits.open("D:\Black hole detection\FITS\MAST_2026-02-20T2123\JWST\jw02736-o001_t001_nircam_clear-f090w\jw02736-o001_t001_nircam_clear-f090w_segm.fits") as hdul:
    image = hdul[1].data.astype(float)
    
results = run_pipeline(image)
visualize_results(image, results)