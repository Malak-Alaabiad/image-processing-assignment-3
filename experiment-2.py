import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numba import njit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images", "lenna.png")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")



# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def psnr(original, processed):
    return cv2.PSNR(original, processed)


# --------------------------------------------------------------------------
# Noise models
# --------------------------------------------------------------------------
def to_uint8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def add_gaussian_noise(img, mean=0, sigma=50):
    """Gaussian noise (mean 0, sigma 50), independent per color channel."""
    np.random.seed(0)
    noise = np.random.normal(mean, sigma, img.shape)
    return to_uint8(img.astype(np.float32) + noise)


def add_salt_pepper_noise(img, amount=0.05, salt_vs_pepper=0.5):
    """Turns a fraction `amount` of pixels fully white (salt) or fully
    black (pepper) on all three channels."""
    noisy = img.copy()
    r = np.random.rand(*img.shape[:2])
    salt = r < amount * salt_vs_pepper
    pepper = (r >= amount * salt_vs_pepper) & (r < amount)
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy


# --------------------------------------------------------------------------
# Spatial filters (OpenCV works on all 3 channels separately)
# --------------------------------------------------------------------------
def kernel(k):
    return np.ones((k, k), np.uint8)


def arithmetic_mean_filter(img, k=3):
    return cv2.blur(img, (k, k))


def median_filter(img, k=3):
    return cv2.medianBlur(img, k)


def max_filter(img, k=3):
    # max over a k x k window = grayscale dilation
    return cv2.dilate(img, kernel(k), borderType=cv2.BORDER_REPLICATE)


def min_filter(img, k=3):
    # min over a k x k window = grayscale erosion
    return cv2.erode(img, kernel(k), borderType=cv2.BORDER_REPLICATE)


def estimate_noise_density(img):
    """Share of values that are pure black (0) or pure white (255).
    Salt & pepper noise sets values to exactly 0 or 255, so this
    estimates how much of the image is noise."""
    return np.mean((img == 0) | (img == 255))


def choose_s_max(img):
    """Biggest window size, picked from how noisy the image is."""
    density = estimate_noise_density(img)
    if density < 0.2:
        return 5
    elif density < 0.4:
        return 7
    elif density < 0.6:
        return 9
    else:
        return 11


def adaptive_median_filter(img, s_max=None):
    """
    Test A: Zmin < Zmed < Zmax ?
        yes -> go to Test B
        no  -> make the window bigger; if it is already s_max, output Zmed
    Test B: Zmin < Zxy < Zmax ?
        yes -> keep Zxy (pixel is not noise)
        no  -> output Zmed
    """
    height, width, channels = img.shape

    # Biggest window size, worked out from the noise density if not given
    if s_max is None:
        s_max = choose_s_max(img)

    # Pad the image so windows near the edges stay inside it
    pad = s_max // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    return adaptive_median_loops(img, padded, s_max, pad)


@njit(cache=True)
def adaptive_median_loops(img, padded, s_max, pad):
    out = img.copy()
    height, width, channels = img.shape

    for c in range(channels):
        for y in range(height):
            for x in range(width):
                zxy = img[y, x, c]

                # Position of this pixel inside the padded image
                py = y + pad
                px = x + pad

                for k in range(3, s_max + 1, 2):  # 3x3, 5x5, 7x7, ...
                    r = k // 2
                    window = padded[py - r:py + r + 1, px - r:px + r + 1, c]

                    z_min = window.min()
                    z_max = window.max()
                    z_med = np.median(window)

                    # Test A: is the median suitable?
                    if z_min < z_med < z_max:
                        # Test B: is the original pixel noise?
                        if z_min < zxy < z_max:
                            out[y, x, c] = zxy      # not noise -> keep it
                        else:
                            out[y, x, c] = z_med    # noise -> replace with median
                        break

                    # Test A failed: try a bigger window,
                    # or use the median if this is already the biggest one
                    if k == s_max:
                        out[y, x, c] = z_med

    return out


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def show_grid(images, titles, suptitle, filename, cols):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.6 * rows),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    original = load_rgb(IMAGE_PATH)

    # ---------------- Part A: Gaussian noise + Adaptive Median ----------------
    gauss = add_gaussian_noise(original, sigma=20)
    gauss_density = estimate_noise_density(gauss)
    gauss_s_max = choose_s_max(gauss)
    gauss_amf = adaptive_median_filter(gauss, s_max=gauss_s_max)
    print(f"Gaussian image: noise density = {gauss_density * 100:.2f}%, "
          f"s_max = {gauss_s_max}")


    # ---------------- Part B: Salt & Pepper + classic filters ----------------
    sp = add_salt_pepper_noise(original, amount=0.05)
    results = {
        "Arithmetic mean 3x3": arithmetic_mean_filter(sp, k=3),
        "Median 3x3": median_filter(sp, k=3),
        "Max 3x3": max_filter(sp, k=3),
        "Min 3x3": min_filter(sp, k=3),
    }

    # ---------------- One figure with both parts ----------------
    # Row 1   : Part A (Gaussian + adaptive median)
    # Row 2-3 : Part B (salt & pepper + mean / median / max / min)
    images = [original, gauss, gauss_amf,
              original, sp] + list(results.values())
    titles = ["Original",
              f"Gaussian noise (σ=20)\nPSNR = {psnr(original, gauss):.2f} dB",
              f"Adaptive median (Smax={gauss_s_max})\nPSNR = {psnr(original, gauss_amf):.2f} dB",
              "Original",
              f"Salt & pepper (5%)\nPSNR = {psnr(original, sp):.2f} dB"]
    titles += [f"{name}\nPSNR = {psnr(original, im):.2f} dB"
               for name, im in results.items()]

    show_grid(images, titles,
              "Experiment 2 - Noise & Restoration Filters (RGB)",
              "exp2_results.png", cols=3)

    # Print a short PSNR summary to the console
    print("\nPSNR vs. original (higher is better)")
    print(f"  Gaussian noisy           : {psnr(original, gauss):6.2f} dB")
    print(f"  Adaptive median          : {psnr(original, gauss_amf):6.2f} dB")
    print(f"  Salt & pepper noisy      : {psnr(original, sp):6.2f} dB")
    for name, im in results.items():
        print(f"  {name:<25}: {psnr(original, im):6.2f} dB")

    plt.show()


if __name__ == "__main__":
    main()
