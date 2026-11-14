"""
Simple GUI for Experiments 1 and 2 (one tab each).

Experiment 1 - Frequency domain filtering (RGB):
    Pick an image and a cutoff D0, then press "Apply".
    Shows the 2x4 grid: image, spectrum, LPF/HPF, filtered spectra, results.

Experiment 2 - Noise & filters (RGB):
    Pick an image, a noise type and a filter, then press "Apply".
    Shows: Original | Noisy | Filtered (with PSNR).
"""

import importlib
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image

try:
    # The file name has a hyphen, so it can't be imported with a plain "import"
    exp2 = importlib.import_module("experiment-2")
    EXP2_ERROR = None
except ImportError as e:  # e.g. opencv / numba not installed
    exp2 = None
    EXP2_ERROR = e

IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif")]

NOISES = ["Gaussian", "Salt & Pepper"]
FILTERS = ["Adaptive median", "Arithmetic mean", "Median", "Max", "Min"]


# --------------------------------------------------------------------------
# Experiment 1 - Frequency domain filtering
# (same steps as experiment-1.py; that script runs on import, so the
#  pipeline is repeated here with D0 and the image taken from the GUI)
# --------------------------------------------------------------------------
EXP1_IMAGE_PATH = Path(__file__).resolve().parent / "images" / "cameraman.png"


def exp1_load(path):
    # Load as RGB -> shape (M, N, 3)
    return np.array(Image.open(path).convert("RGB"), dtype=float)


def exp1_items(img, D0):
    M, N = img.shape[:2]

    # 1. DFT -> frequency domain (shift so low frequencies are at the center)
    #    axes=(0, 1): one 2D DFT per color channel (R, G, B)
    F = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))

    # Distance of every frequency from the center
    u, v = np.meshgrid(np.arange(N) - N // 2, np.arange(M) - M // 2)
    D = np.sqrt(u**2 + v**2)

    # 2/3. Gaussian low-pass and high-pass filters (cutoff D0)
    LPF = np.exp(-(D**2) / (2 * (D0**2)))
    HPF = 1.0 - LPF

    # Apply filters in the frequency domain (same mask on all 3 channels)
    F_lpf = F * LPF[..., None]
    F_hpf = F * HPF[..., None]

    # 4. Back to the spatial domain (Inverse DFT)
    img_lpf = np.real(np.fft.ifft2(np.fft.ifftshift(F_lpf, axes=(0, 1)), axes=(0, 1)))
    img_hpf = np.real(np.fft.ifft2(np.fft.ifftshift(F_hpf, axes=(0, 1)), axes=(0, 1)))

    # Spectra are shown as the mean of the R, G, B spectra
    spec = lambda X: np.log1p(np.abs(X)).mean(axis=2)
    # RGB images must be in 0..1 for imshow
    rescale = lambda x: (x - x.min()) / (x.max() - x.min())

    # 5. Visualization items (2x4 grid)
    return [
        (img / 255, "Original image", {}),
        (spec(F), "Original spectrum (DFT)", {}),
        (LPF, f"Low-pass filter (D0={D0})", dict(vmin=0, vmax=1)),
        (HPF, f"High-pass filter (D0={D0})", dict(vmin=0, vmax=1)),
        (spec(F_lpf), "Spectrum after LPF", {}),
        (spec(F_hpf), "Spectrum after HPF", {}),
        (np.clip(img_lpf, 0, 255) / 255, "Spatial domain: LPF (blurred)", {}),
        (rescale(img_hpf), "Spatial domain: HPF (edges)", {}),
    ]


class Exp1Tab:
    def __init__(self, parent):
        self.image = exp1_load(EXP1_IMAGE_PATH)

        # ---------------- Controls ----------------
        controls = ttk.Frame(parent, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="Load image...", command=self.load_image).pack(fill=tk.X, pady=(0, 15))

        ttk.Label(controls, text="Cutoff frequency D0").pack(anchor=tk.W)
        self.d0 = tk.IntVar(value=30)
        tk.Scale(controls, from_=1, to=150, orient=tk.HORIZONTAL, variable=self.d0).pack(fill=tk.X)

        ttk.Button(controls, text="Apply", command=self.apply).pack(fill=tk.X, pady=(20, 0))

        self.status = ttk.Label(controls, text="", wraplength=200)
        self.status.pack(anchor=tk.W, pady=(10, 0))

        # ---------------- Images ----------------
        self.figure = Figure(figsize=(14, 7), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.apply()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if path:
            self.image = exp1_load(path)
            self.apply()

    def apply(self):
        D0 = self.d0.get()

        self.figure.clear()
        for i, (im, title, kwargs) in enumerate(exp1_items(self.image, D0)):
            ax = self.figure.add_subplot(2, 4, i + 1)
            ax.imshow(im, cmap="gray", **kwargs)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        self.canvas.draw()

        M, N = self.image.shape[:2]
        self.status.config(text=f"Image size: {M} x {N} (RGB)")


# --------------------------------------------------------------------------
# Experiment 2 - Noise & restoration filters
# --------------------------------------------------------------------------
class Exp2Tab:
    def __init__(self, parent):
        # Fall back to Lenna.png next to the script if images/ doesn't exist
        path = exp2.IMAGE_PATH
        if not os.path.exists(path):
            path = os.path.join(exp2.BASE_DIR, "Lenna.png")
        self.image = exp2.load_rgb(path)

        # ---------------- Controls ----------------
        controls = ttk.Frame(parent, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(controls, text="Load image...", command=self.load_image).pack(fill=tk.X, pady=(0, 15))

        ttk.Label(controls, text="Noise").pack(anchor=tk.W)
        self.noise = tk.StringVar(value=NOISES[0])
        for name in NOISES:
            ttk.Radiobutton(controls, text=name, value=name, variable=self.noise).pack(anchor=tk.W)

        ttk.Label(controls, text="Gaussian sigma").pack(anchor=tk.W, pady=(10, 0))
        self.sigma = tk.IntVar(value=20)
        tk.Scale(controls, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.sigma).pack(fill=tk.X)

        ttk.Label(controls, text="Salt & pepper amount (%)").pack(anchor=tk.W, pady=(10, 0))
        self.amount = tk.IntVar(value=5)
        tk.Scale(controls, from_=1, to=90, orient=tk.HORIZONTAL, variable=self.amount).pack(fill=tk.X)

        ttk.Label(controls, text="Filter").pack(anchor=tk.W, pady=(15, 0))
        self.filter = tk.StringVar(value=FILTERS[0])
        filter_box = ttk.Combobox(controls, values=FILTERS, textvariable=self.filter, state="readonly")
        filter_box.pack(fill=tk.X)
        # Picking a filter selects its matching noise automatically
        filter_box.bind("<<ComboboxSelected>>", lambda e: self.match_noise())
        self.match_noise()

        ttk.Label(controls, text="Kernel size (not for adaptive)").pack(anchor=tk.W, pady=(10, 0))
        self.ksize = tk.IntVar(value=3)
        ttk.Combobox(controls, values=[3, 5, 7, 9], textvariable=self.ksize, state="readonly").pack(fill=tk.X)

        ttk.Button(controls, text="Apply", command=self.apply).pack(fill=tk.X, pady=(20, 0))

        self.status = ttk.Label(controls, text="", wraplength=200)
        self.status.pack(anchor=tk.W, pady=(10, 0))

        # ---------------- Images ----------------
        self.figure = Figure(figsize=(12, 4.5), layout="constrained")
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.apply()

    def match_noise(self):
        """Adaptive median -> salt & pepper, every other filter -> Gaussian."""
        if self.filter.get() == "Adaptive median":
            self.noise.set("Salt & Pepper")
        else:
            self.noise.set("Gaussian")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=IMAGE_TYPES)
        if path:
            self.image = exp2.load_rgb(path)
            self.apply()

    def apply(self):
        original = self.image

        # Add noise
        if self.noise.get() == "Gaussian":
            noisy = exp2.add_gaussian_noise(original, sigma=self.sigma.get())
            noise_title = f"Gaussian noise (σ={self.sigma.get()})"
        else:
            noisy = exp2.add_salt_pepper_noise(original, amount=self.amount.get() / 100)
            noise_title = f"Salt & pepper ({self.amount.get()}%)"

        # Apply filter
        name = self.filter.get()
        k = self.ksize.get()
        if name == "Adaptive median":
            s_max = exp2.choose_s_max(noisy)
            filtered = exp2.adaptive_median_filter(noisy, s_max=s_max)
            filter_title = f"Adaptive median (Smax={s_max})"
        elif name == "Arithmetic mean":
            filtered = exp2.arithmetic_mean_filter(noisy, k)
            filter_title = f"Arithmetic mean {k}x{k}"
        elif name == "Median":
            filtered = exp2.median_filter(noisy, k)
            filter_title = f"Median {k}x{k}"
        elif name == "Max":
            filtered = exp2.max_filter(noisy, k)
            filter_title = f"Max {k}x{k}"
        else:
            filtered = exp2.min_filter(noisy, k)
            filter_title = f"Min {k}x{k}"

        # Show the three images
        images = [original, noisy, filtered]
        titles = ["Original",
                  f"{noise_title}\nPSNR = {exp2.psnr(original, noisy):.2f} dB",
                  f"{filter_title}\nPSNR = {exp2.psnr(original, filtered):.2f} dB"]

        self.figure.clear()
        for i, (im, title) in enumerate(zip(images, titles)):
            ax = self.figure.add_subplot(1, 3, i + 1)
            ax.imshow(im)
            ax.set_title(title)
            ax.axis("off")
        self.canvas.draw()

        self.status.config(text=f"Noise density (0/255 values): "
                                f"{exp2.estimate_noise_density(noisy) * 100:.2f}%")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.title("Image Processing - Assignment 3")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    tab1 = ttk.Frame(notebook)
    notebook.add(tab1, text="Experiment 1 - Frequency filtering")
    Exp1Tab(tab1)

    tab2 = ttk.Frame(notebook)
    notebook.add(tab2, text="Experiment 2 - Noise & filters")
    if exp2 is not None:
        Exp2Tab(tab2)
    else:
        ttk.Label(tab2, padding=20,
                  text=f"Experiment 2 is unavailable: {EXP2_ERROR}\n"
                       f"Install the missing package (e.g. pip install opencv-python numba).").pack()

    root.mainloop()


if __name__ == "__main__":
    main()
