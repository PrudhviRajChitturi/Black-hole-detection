import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import cv2
from PIL import Image, ImageTk
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import threading

from run_pipeline import run_pipeline, visualize_results


# ----------------------------
# Physics
# ----------------------------

G = 6.67430e-11
c = 299792458
solar_mass = 1.98847e30

def compute_physics(theta_E_arcsec, z_l, z_s):

    theta_rad = np.deg2rad(theta_E_arcsec / 3600)

    D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
    D_s = cosmo.angular_diameter_distance(z_s).to(u.m).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value

    mass = (c**2 / (4 * G)) * theta_rad**2 * (D_l * D_s / D_ls)
    mass_solar = mass / solar_mass

    ring_size = theta_rad * D_l
    Rs = 2 * G * mass / c**2

    velocity = 200_000  # m/s
    timescale = ring_size / velocity

    return mass_solar, ring_size, Rs, timescale


# ----------------------------
# UI App
# ----------------------------

class App:

    def __init__(self, root):

        self.root = root
        self.root.title("Einstein Ring Detector")

        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        control = tk.Frame(root)
        control.pack()

        tk.Button(control, text="Load Image", command=self.load_image).grid(row=0, column=0)
        
        self.status_label = tk.Label(self.root, text="Ready", anchor="w")
        self.status_label.pack(fill="x", pady=5)

        tk.Label(control, text="Lens z:").grid(row=0, column=1)
        self.z_l = tk.Entry(control, width=6)
        self.z_l.insert(0, "0.5")
        self.z_l.grid(row=0, column=2)

        tk.Label(control, text="Source z:").grid(row=0, column=3)
        self.z_s = tk.Entry(control, width=6)
        self.z_s.insert(0, "2.0")
        self.z_s.grid(row=0, column=4)

        self.output = tk.Text(root, height=15)
        self.output.pack(fill="both")

        self.image = None
        
        self.progress = ttk.Progressbar(
            self.root,
            mode="indeterminate",
            length=300
        )
        self.progress.pack(pady=5)

    # ----------------------------

    def load_image(self):

        path = filedialog.askopenfilename()

        if path.endswith(".npy"):
            img = np.load(path)
            if img.ndim == 3:
                print("Multi-channel detected, using channel 0.")
                img = img[..., 0]
            elif img.ndim != 2:
                raise ValueError("Unsupported image shape")
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(float)
        self.image = img

        self.display_image(img)
        self.status_label.config(text="Image loaded. Starting analysis.......")
        threading.Thread(target=self.run_analysis, daemon=True).start()
        

    # ----------------------------

    def display_image(self, img):

        norm = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img8 = (norm * 255).astype(np.uint8)

        pil_img = Image.fromarray(img8)
        pil_img = pil_img.resize((600, 600))

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

    # ----------------------------

    def run_analysis(self):
        
        if self.image is None:
            return
        
        self.root.after(0, self.progress.start)
        self.root.after(0, lambda:
            self.status_label.config(text="Analyzing Image......")
        )

        results = run_pipeline(self.image)

        self.root.after(0, self.progress.stop)
        self.root.after(0, lambda:
            self.status_label.config(text="Analysis complete.")
        )

        self.output.delete("1.0", tk.END)

        z_l = float(self.z_l.get())
        z_s = float(self.z_s.get())

        if len(results) == 0:
            self.output.insert(tk.END, "No candidates detected.\n")
            return

        for i, obj in enumerate(results):

            prob = obj["prob"]
            theta_E = obj["stats"]["theta_E"]

            mass, ring_size, Rs, timescale = compute_physics(theta_E, z_l, z_s)

            self.output.insert(tk.END, f"\nCandidate {i+1}\n")
            self.output.insert(tk.END, f"P(BH): {prob-0.06:.3f}\n")
            self.output.insert(tk.END, f"Einstein Radius: {theta_E:.5f} arcsec\n")
            self.output.insert(tk.END, f"Mass: {mass:.2e} M☉\n")
            self.output.insert(tk.END, f"Ring Size: {ring_size:.2e} m\n")
            self.output.insert(tk.END, f"Schwarzschild Radius: {Rs:.2e} m\n")
            self.output.insert(tk.END, f"Timescale: {timescale:.2e} s\n")


# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":

    root = tk.Tk()
    app = App(root)
    root.mainloop()