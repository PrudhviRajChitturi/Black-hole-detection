import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u




# ----------------------------
# Physics
# ----------------------------

G = 6.67430e-11
c = 299792458
solar_mass = 1.98847e30
AU = 1.496e+11

for i in range (0,10):

    z_l = float(input("enter z_l: "))
    z_s = float(input("enter z_s: "))
    theta_e = float(input("enter theta_e: "))

    theta_rad = np.deg2rad(theta_e / 3600)

    D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
    D_s = cosmo.angular_diameter_distance(z_s).to(u.m).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value

    mass = (c**2 / (4 * G)) * theta_rad**2 * (D_l * D_s / D_ls)
    mass_solar = mass / solar_mass

    ring_size = theta_rad * D_l
    Rs = 2 * G * mass / c**2
    blr = Rs / AU

    velocity = 200_000  # m/s
    timescale = ring_size / velocity
    
    

    print(f"Mass: {mass_solar:.2e} M☉\n")
    print(f"Ring Size: {ring_size:.2e} m\n")
    print(f"Schwarzschild Radius: {Rs:.2e} m or {blr:.3f} AU\n")
    print(f"Timescale: {timescale:.2e} s\n")





