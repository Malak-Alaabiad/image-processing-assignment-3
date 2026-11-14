import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

script_dir = Path(__file__).resolve().parent
img_path = script_dir / "images" / "cameraman.png"

# Load the image as RGB -> shape (M, N, 3)
img = np.array(Image.open(img_path).convert("RGB"), dtype=float)
M, N = img.shape[:2]
print(f"Loaded image '{img_path.name}' of shape ({M}, {N})")

# 1. DFT -> frequency domain (shift so low frequencies are at the center)
#    axes=(0, 1): one 2D DFT per color channel (R, G, B)
F = np.fft.fftshift(np.fft.fft2(img, axes=(0, 1)), axes=(0, 1))

# Distance of every frequency from the center
u, v = np.meshgrid(np.arange(N) - N // 2, np.arange(M) - M // 2)
D = np.sqrt(u**2 + v**2)

# 2/3. Gaussian low-pass and high-pass filters (cutoff D0)
D0 = 30
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

# 5. Visualization (2x4 grid as required by Lab 4)
items = [
    (img / 255, "Original image", {}),
    (spec(F), "Original spectrum (DFT)", {}),
    (LPF, f"Low-pass filter (D0={D0})", dict(vmin=0, vmax=1)),
    (HPF, f"High-pass filter (D0={D0})", dict(vmin=0, vmax=1)),
    (spec(F_lpf), "Spectrum after LPF", {}),
    (spec(F_hpf), "Spectrum after HPF", {}),
    (np.clip(img_lpf, 0, 255) / 255, "Spatial domain: LPF (blurred)", {}),
    (rescale(img_hpf), "Spatial domain: HPF (edges)", {}),
]

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for ax, (im, t, kwargs) in zip(axs.ravel(), items):
    ax.imshow(im, cmap="gray", **kwargs)
    ax.set_title(t, fontsize=11)
    ax.axis("off")

plt.tight_layout()
out_dir = script_dir / "outputs"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / "exp1_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Results successfully saved to: {out_path.relative_to(script_dir)}")

# Show the interactive plot window if supported
if matplotlib.get_backend().lower() != "agg":
    plt.show()
