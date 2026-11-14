"""
GUI for Assignment 3 - one page per experiment, switched from the header.

Frequency filtering (Experiment 1):
    Load an image and drag the cutoff D0 slider. The 2x4 grid (image,
    spectrum, LPF/HPF, filtered spectra and results) updates live.

Noise & filters (Experiment 2):
    Load an image, pick a noise type and a filter, then press "Apply".
    Shows Original | Noisy | Filtered, with PSNR scores above the images.

Shortcuts: Ctrl+O load image, Ctrl+S save figure, Enter apply.
"""

import importlib
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import numpy as np
from matplotlib import font_manager
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

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")]

NOISES = ["Gaussian", "Salt & Pepper"]
FILTERS = ["Adaptive median", "Arithmetic mean", "Median", "Max", "Min"]
KERNEL_SIZES = [3, 5, 7, 9]

# --------------------------------------------------------------------------
# Colors (dark slate theme with an indigo accent)
# --------------------------------------------------------------------------
BG = "#0f172a"            # window background
SURFACE = "#1e293b"       # cards
SURFACE_2 = "#334155"     # controls and tiles on a card
HOVER = "#475569"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DISABLED = "#64748b"
ACCENT = "#6366f1"
ACCENT_HOVER = "#818cf8"
ACCENT_PRESSED = "#4f46e5"
GOOD = "#4ade80"
BAD = "#f87171"


# --------------------------------------------------------------------------
# Experiment 1 - Frequency domain filtering
# (same steps as experiment-1.py; that script runs on import, so the
#  pipeline is repeated here with D0 and the image taken from the GUI)
# --------------------------------------------------------------------------
EXP1_IMAGE_PATH = IMAGES_DIR / "cameraman.png"


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


def load_rgb(path):
    """Load any image file as an RGB uint8 array of shape (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


# --------------------------------------------------------------------------
# Window setup and theme
# --------------------------------------------------------------------------
def enable_hidpi():
    """Render at full resolution on Windows displays scaled above 100%
    (otherwise Windows stretches the window and text looks blurry)."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass  # not Windows, or an old version


def dark_title_bar(root):
    """Ask Windows 10/11 for a dark title bar that matches the theme."""
    try:
        import ctypes
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        set_attr = ctypes.windll.dwmapi.DwmSetWindowAttribute
        on = ctypes.c_int(1)
        for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE (newer, older builds)
            if set_attr(hwnd, attr, ctypes.byref(on), ctypes.sizeof(on)) == 0:
                break
        # Windows 11 only: paint the title bar with the exact background color
        r, g, b = (int(BG[i:i + 2], 16) for i in (1, 3, 5))
        color = ctypes.c_int(r | g << 8 | b << 16)
        set_attr(hwnd, 35, ctypes.byref(color), ctypes.sizeof(color))
    except Exception:
        pass


class Theme:
    """Fonts, pixel scaling and ttk styles shared by the whole app."""

    def __init__(self, root):
        self.scale = root.winfo_fpixels("1i") / 96  # 1.0 at 100% display scaling
        family = self._pick_family(root)
        self.mpl_family = self._pick_mpl_family(family)
        self.fonts = {
            "base": (family, 10),
            "small": (family, 9),
            "bold": (family, 10, "bold"),
            "section": (family, 8, "bold"),
            "title": (family, 16, "bold"),
            "metric": (family, 16, "bold"),
        }
        self._configure_styles(root)

    def px(self, n):
        """Scale a size in pixels to the display's DPI."""
        return int(round(n * self.scale))

    @staticmethod
    def _pick_family(root):
        available = set(tkfont.families(root))
        for name in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Ubuntu", "Cantarell"):
            if name in available:
                return name
        return tkfont.nametofont("TkDefaultFont").actual("family")

    @staticmethod
    def _pick_mpl_family(family):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
            return family
        except ValueError:
            return "DejaVu Sans"

    def _configure_styles(self, root):
        px, f = self.px, self.fonts
        root.configure(bg=BG)
        style = ttk.Style(root)
        style.theme_use("clam")  # the built-in theme that allows full recoloring

        style.configure(".", background=SURFACE, foreground=TEXT, font=f["base"],
                        bordercolor=SURFACE, lightcolor=SURFACE, darkcolor=SURFACE,
                        troughcolor=SURFACE_2, focuscolor=ACCENT,
                        selectbackground=ACCENT, selectforeground="white")

        # Frames
        style.configure("App.TFrame", background=BG)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("Seg.TFrame", background=SURFACE_2)
        style.configure("Nav.TFrame", background=SURFACE)
        style.configure("Tile.TFrame", background=SURFACE_2)

        # Labels
        style.configure("TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED, font=f["small"])
        style.configure("Section.TLabel", foreground=MUTED, font=f["section"])
        style.configure("Value.TLabel", foreground=ACCENT_HOVER, font=f["bold"])
        style.configure("Title.TLabel", background=BG, font=f["title"])
        style.configure("Subtitle.TLabel", background=BG, foreground=MUTED)
        style.configure("Status.TLabel", background=BG, foreground=MUTED, font=f["small"])
        style.configure("TileName.TLabel", background=SURFACE_2, foreground=MUTED, font=f["small"])
        for name, color in (("TileValue", TEXT), ("TileGood", GOOD), ("TileBad", BAD)):
            style.configure(f"{name}.TLabel", background=SURFACE_2, foreground=color,
                            font=f["metric"])

        def flat(name, element, bg, fg, bg_states, fg_states, **options):
            """A borderless button style: no focus ring, borders drawn in the fill color."""
            style.layout(name, [(f"{element}.border", {"sticky": "nswe", "children": [
                (f"{element}.padding", {"sticky": "nswe", "children": [
                    (f"{element}.label", {"sticky": "nswe"})]})]})])
            style.configure(name, background=bg, foreground=fg, borderwidth=0, relief="flat",
                            bordercolor=bg, lightcolor=bg, darkcolor=bg, **options)
            style.map(name, background=bg_states, bordercolor=bg_states,
                      lightcolor=bg_states, darkcolor=bg_states, foreground=fg_states,
                      relief=[("pressed", "flat"), ("selected", "flat"), ("active", "flat")])

        # Buttons
        flat("TButton", "Button", SURFACE_2, TEXT,
             [("disabled", SURFACE_2), ("pressed", SURFACE_2), ("active", HOVER)],
             [("disabled", DISABLED)],
             padding=(px(12), px(7)), anchor="center")
        flat("Accent.TButton", "Button", ACCENT, "white",
             [("disabled", HOVER), ("pressed", ACCENT_PRESSED), ("active", ACCENT_HOVER)],
             [("disabled", MUTED)],
             padding=(px(12), px(9)), anchor="center", font=f["bold"])

        # Radio buttons drawn as segmented controls, list items and page tabs
        flat("Seg.Toolbutton", "Toolbutton", SURFACE_2, MUTED,
             [("disabled", SURFACE_2), ("selected", ACCENT), ("active", HOVER)],
             [("disabled", DISABLED), ("selected", "white"), ("active", TEXT)],
             padding=(px(6), px(6)), anchor="center")
        flat("List.Toolbutton", "Toolbutton", SURFACE, TEXT,
             [("selected", ACCENT), ("active", SURFACE_2)],
             [("selected", "white")],
             padding=(px(10), px(6)), anchor="w")
        flat("Nav.Toolbutton", "Toolbutton", SURFACE, MUTED,
             [("selected", ACCENT), ("active", SURFACE_2)],
             [("selected", "white"), ("active", TEXT)],
             padding=(px(16), px(7)), anchor="center", font=f["bold"])

        # Sliders
        style.configure("Horizontal.TScale", background=ACCENT, troughcolor=SURFACE_2,
                        bordercolor=SURFACE_2, lightcolor=ACCENT, darkcolor=ACCENT,
                        gripcount=0, sliderlength=px(18), sliderthickness=px(16))
        style.map("Horizontal.TScale",
                  background=[("active", ACCENT_HOVER)],
                  lightcolor=[("active", ACCENT_HOVER)],
                  darkcolor=[("active", ACCENT_HOVER)])


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
class Page:
    """Base page: a control sidebar on the left and a figure card on the right."""

    def __init__(self, app, parent):
        self.app = app
        self.px = app.theme.px
        self.status = ""
        self.frame = ttk.Frame(parent, style="App.TFrame")

    # ---- Keyboard / menu actions (overridden by pages that support them) ----
    def load_image(self):
        pass

    def save_figure(self):
        pass

    def apply(self):
        pass

    def default_filename(self):
        return "figure.png"

    # ---- Helpers ----
    def set_status(self, text):
        self.status = text
        if self.app.current is self:
            self.app.status.config(text=text)

    def build_layout(self):
        px = self.px
        self.frame.columnconfigure(1, weight=1)
        self.frame.columnconfigure(0, minsize=px(320))  # same sidebar width on every page
        self.frame.rowconfigure(0, weight=1)
        self.sidebar = ttk.Frame(self.frame, style="Card.TFrame", padding=px(18))
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, px(14)))
        self.main = ttk.Frame(self.frame, style="Card.TFrame", padding=px(14))
        self.main.grid(row=0, column=1, sticky="nsew")

    def section(self, text, first=False):
        ttk.Label(self.sidebar, text=text.upper(), style="Section.TLabel").pack(
            anchor=tk.W, pady=(0 if first else self.px(16), self.px(8)))

    def note(self, parent, text=""):
        return ttk.Label(parent, text=text, style="Muted.TLabel",
                         wraplength=self.px(262), justify=tk.LEFT)

    def slider(self, parent, label, var, lo, hi, suffix="", on_change=None):
        """A labelled slider that snaps to whole numbers and shows its value."""
        box = ttk.Frame(parent, style="Card.TFrame")
        head = ttk.Frame(box, style="Card.TFrame")
        head.pack(fill=tk.X)
        ttk.Label(head, text=label).pack(side=tk.LEFT)
        value = ttk.Label(head, text=f"{var.get()}{suffix}", style="Value.TLabel")
        value.pack(side=tk.RIGHT)
        last = [var.get()]

        def written(*_):
            v = var.get()
            value.config(text=f"{v}{suffix}")
            if v != last[0]:
                last[0] = v
                if on_change:
                    on_change()

        var.trace_add("write", written)
        ttk.Scale(box, from_=lo, to=hi, orient=tk.HORIZONTAL, variable=var,
                  command=lambda raw: var.set(int(round(float(raw))))).pack(
            fill=tk.X, pady=(self.px(6), 0))
        return box

    def tile(self, parent, column, name):
        """A metric tile (small caption + big value); returns the value label."""
        tile = ttk.Frame(parent, style="Tile.TFrame", padding=(self.px(14), self.px(10)))
        tile.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else self.px(10), 0))
        parent.columnconfigure(column, weight=1, uniform="tiles")
        ttk.Label(tile, text=name, style="TileName.TLabel").pack(anchor=tk.W)
        value = ttk.Label(tile, text="–", style="TileValue.TLabel")
        value.pack(anchor=tk.W)
        return value

    def add_figure(self, parent):
        self.figure = Figure(facecolor=SURFACE, layout="constrained")
        self.figure.get_layout_engine().set(w_pad=6 / 72, h_pad=6 / 72, wspace=0.03, hspace=0.05)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        widget = self.canvas.get_tk_widget()
        widget.configure(background=SURFACE, highlightthickness=0, borderwidth=0)
        widget.pack(fill=tk.BOTH, expand=True)

    def draw(self, panels, rows, cols):
        """panels: list of (image, title, imshow kwargs)."""
        self.figure.clear()
        for i, (img, title, kwargs) in enumerate(panels):
            ax = self.figure.add_subplot(rows, cols, i + 1)
            ax.imshow(img, **kwargs)
            ax.set_title(title, color=TEXT, fontsize=10, pad=8,
                         fontfamily=self.app.theme.mpl_family)
            ax.axis("off")
        self.canvas.draw_idle()

    def ask_image(self):
        return filedialog.askopenfilename(parent=self.app.root, title="Load image",
                                          initialdir=IMAGES_DIR, filetypes=IMAGE_TYPES)

    def save_dialog(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.app.root, title="Save figure", initialdir=OUTPUT_DIR,
            initialfile=self.default_filename(), defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg")])
        if path:
            self.figure.savefig(path, dpi=150, facecolor=self.figure.get_facecolor())
            self.set_status(f"Saved {Path(path).name}")


class FrequencyPage(Page):
    """Experiment 1 - Gaussian low/high-pass filtering in the frequency domain."""

    def __init__(self, app, parent):
        super().__init__(app, parent)
        px = self.px
        self.build_layout()
        self.image_path = EXP1_IMAGE_PATH
        self.image = exp1_load(self.image_path)
        self._pending = None

        # Bottom actions are packed first so they never get squeezed out
        ttk.Button(self.sidebar, text="Save figure…", command=self.save_figure).pack(
            side=tk.BOTTOM, fill=tk.X)

        self.section("Image", first=True)
        ttk.Button(self.sidebar, text="Load image…", command=self.load_image).pack(fill=tk.X)
        self.info = self.note(self.sidebar)
        self.info.pack(anchor=tk.W, pady=(px(8), 0))

        self.section("Gaussian filter")
        self.d0 = tk.IntVar(value=30)
        self.slider(self.sidebar, "Cutoff frequency D0", self.d0, 1, 150,
                    on_change=self.schedule_refresh).pack(fill=tk.X)
        self.note(self.sidebar,
                  "Low-pass keeps low frequencies (smooth areas); high-pass keeps "
                  "high frequencies (edges and fine detail). A larger D0 lets more "
                  "through the low-pass.\n\nThe preview updates as you drag.").pack(
            anchor=tk.W, pady=(px(14), 0))

        self.add_figure(self.main)
        self.refresh()

    def schedule_refresh(self):
        # Wait for a short pause while dragging, so the grid doesn't redraw on every pixel
        if self._pending:
            self.frame.after_cancel(self._pending)
        self._pending = self.frame.after(120, self.refresh)

    def refresh(self):
        self._pending = None
        start = time.perf_counter()
        D0 = self.d0.get()
        panels = [(img, title, dict(cmap="gray", **kwargs))
                  for img, title, kwargs in exp1_items(self.image, D0)]
        self.draw(panels, rows=2, cols=4)
        elapsed = (time.perf_counter() - start) * 1000

        M, N = self.image.shape[:2]
        self.info.config(text=f"{self.image_path.name}  ·  {N} × {M} px")
        self.set_status(f"Frequency filtering  ·  D0 = {D0}  ·  computed in {elapsed:.0f} ms")

    def load_image(self):
        path = self.ask_image()
        if not path:
            return
        try:
            self.image = exp1_load(path)
        except Exception as e:
            messagebox.showerror("Could not open image", str(e), parent=self.app.root)
            return
        self.image_path = Path(path)
        self.refresh()

    def apply(self):
        self.refresh()

    def save_figure(self):
        self.save_dialog()

    def default_filename(self):
        return f"exp1_D0-{self.d0.get()}.png"


class NoisePage(Page):
    """Experiment 2 - add noise, restore it with a filter and compare PSNR."""

    def __init__(self, app, parent):
        super().__init__(app, parent)
        px = self.px
        self.build_layout()
        self.image_path = Path(exp2.IMAGE_PATH)
        self.image = load_rgb(self.image_path)
        self.busy = False

        # Bottom actions are packed first so they never get squeezed out
        actions = ttk.Frame(self.sidebar, style="Card.TFrame")
        actions.pack(side=tk.BOTTOM, fill=tk.X, pady=(px(12), 0))
        actions.columnconfigure(0, weight=1)
        self.apply_button = ttk.Button(actions, text="Apply", style="Accent.TButton",
                                       command=self.apply)
        self.apply_button.grid(row=0, column=0, sticky="nsew")
        ttk.Button(actions, text="Save figure…", command=self.save_figure).grid(
            row=0, column=1, sticky="nsew", padx=(px(8), 0))

        # Image
        self.section("Image", first=True)
        ttk.Button(self.sidebar, text="Load image…", command=self.load_image).pack(fill=tk.X)
        self.info = self.note(self.sidebar)
        self.info.pack(anchor=tk.W, pady=(px(8), 0))

        # Noise: segmented choice + the slider that belongs to it
        self.section("Noise")
        self.noise = tk.StringVar(value=NOISES[0])
        bar, _ = app.segmented(self.sidebar, self.noise, [(n, n) for n in NOISES],
                               command=self.refresh_controls)
        bar.pack(fill=tk.X)
        self.sigma = tk.IntVar(value=20)
        self.amount = tk.IntVar(value=5)
        params = ttk.Frame(self.sidebar, style="Card.TFrame")
        params.pack(fill=tk.X, pady=(px(12), 0))
        self.sigma_box = self.slider(params, "Gaussian σ", self.sigma, 1, 100)
        self.amount_box = self.slider(params, "Amount", self.amount, 1, 90, suffix=" %")

        # Filter: list of choices + kernel size
        self.section("Filter")
        self.filter = tk.StringVar(value=FILTERS[0])
        for name in FILTERS:
            ttk.Radiobutton(self.sidebar, text=name, value=name, variable=self.filter,
                            style="List.Toolbutton", takefocus=False,
                            command=self.on_filter_change).pack(fill=tk.X, pady=(0, px(2)))
        head = ttk.Frame(self.sidebar, style="Card.TFrame")
        head.pack(fill=tk.X, pady=(px(10), px(6)))
        ttk.Label(head, text="Kernel size").pack(side=tk.LEFT)
        self.kernel_hint = ttk.Label(head, style="Muted.TLabel")
        self.kernel_hint.pack(side=tk.RIGHT)
        self.ksize = tk.IntVar(value=3)
        bar, self.kernel_buttons = app.segmented(
            self.sidebar, self.ksize, [(k, f"{k}×{k}") for k in KERNEL_SIZES])
        bar.pack(fill=tk.X)

        # Results: metric tiles above the images
        tiles = ttk.Frame(self.main, style="Card.TFrame")
        tiles.pack(fill=tk.X, pady=(0, px(6)))
        self.noisy_value = self.tile(tiles, 0, "Noisy PSNR")
        self.filtered_value = self.tile(tiles, 1, "Filtered PSNR")
        self.gain_value = self.tile(tiles, 2, "Improvement")
        self.density_value = self.tile(tiles, 3, "Noise density (0/255 values)")
        self.add_figure(self.main)

        self.match_noise()
        self.refresh_controls()
        self.update_info()
        self.frame.after(50, self.apply)  # first run once the window is up

    # ---- Controls ----
    def match_noise(self):
        """Adaptive median -> salt & pepper, every other filter -> Gaussian."""
        if self.filter.get() == "Adaptive median":
            self.noise.set("Salt & Pepper")
        else:
            self.noise.set("Gaussian")

    def on_filter_change(self):
        self.match_noise()
        self.refresh_controls()

    def refresh_controls(self):
        """Show only the controls that apply to the current choices."""
        gaussian = self.noise.get() == "Gaussian"
        self.sigma_box.pack_forget()
        self.amount_box.pack_forget()
        (self.sigma_box if gaussian else self.amount_box).pack(fill=tk.X)

        adaptive = self.filter.get() == "Adaptive median"
        for button in self.kernel_buttons:
            button.state(["disabled"] if adaptive else ["!disabled"])
        self.kernel_hint.config(text="set automatically (S_max)" if adaptive else "")

    def update_info(self):
        H, W = self.image.shape[:2]
        self.info.config(text=f"{self.image_path.name}  ·  {W} × {H} px")

    # ---- Actions ----
    def load_image(self):
        path = self.ask_image()
        if not path:
            return
        try:
            self.image = load_rgb(path)
        except Exception as e:
            messagebox.showerror("Could not open image", str(e), parent=self.app.root)
            return
        self.image_path = Path(path)
        self.update_info()
        self.apply()

    def apply(self):
        if self.busy:
            return
        self.busy = True
        root = self.app.root
        self.apply_button.state(["disabled"])
        self.apply_button.config(text="Processing…")
        root.config(cursor="watch")
        self.set_status("Processing…")
        root.update_idletasks()
        try:
            self.process()
        finally:
            self.busy = False
            self.apply_button.state(["!disabled"])
            self.apply_button.config(text="Apply")
            root.config(cursor="")

    def process(self):
        start = time.perf_counter()
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
        else:
            filter_func = {
                "Arithmetic mean": exp2.arithmetic_mean_filter,
                "Median": exp2.median_filter,
                "Max": exp2.max_filter,
                "Min": exp2.min_filter,
            }[name]
            filtered = filter_func(noisy, k)
            filter_title = f"{name} {k}×{k}"
        elapsed = time.perf_counter() - start

        # Scores
        noisy_psnr = exp2.psnr(original, noisy)
        filtered_psnr = exp2.psnr(original, filtered)
        gain = filtered_psnr - noisy_psnr
        self.noisy_value.config(text=f"{noisy_psnr:.2f} dB")
        self.filtered_value.config(text=f"{filtered_psnr:.2f} dB")
        self.gain_value.config(text=f"{gain:+.2f} dB",
                               style="TileGood.TLabel" if gain >= 0 else "TileBad.TLabel")
        self.density_value.config(text=f"{exp2.estimate_noise_density(noisy) * 100:.1f} %")

        # Images
        self.draw([
            (original, "Original", {}),
            (noisy, f"{noise_title}\nPSNR = {noisy_psnr:.2f} dB", {}),
            (filtered, f"{filter_title}\nPSNR = {filtered_psnr:.2f} dB", {}),
        ], rows=1, cols=3)
        self.set_status(f"{filter_title} on {noise_title}  ·  processed in {elapsed:.2f} s")

    def save_figure(self):
        self.save_dialog()

    def default_filename(self):
        noise = "gaussian" if self.noise.get() == "Gaussian" else "salt-pepper"
        filt = self.filter.get().lower().replace(" ", "-")
        return f"exp2_{noise}_{filt}.png"


class UnavailablePage(Page):
    """Shown instead of Experiment 2 when its packages are missing."""

    def __init__(self, app, parent):
        super().__init__(app, parent)
        card = ttk.Frame(self.frame, style="Card.TFrame", padding=self.px(28))
        card.place(relx=0.5, rely=0.4, anchor="center")
        ttk.Label(card, text="Experiment 2 is unavailable",
                  font=app.theme.fonts["title"]).pack(anchor=tk.W)
        ttk.Label(card, style="Muted.TLabel", justify=tk.LEFT,
                  text=f"{EXP2_ERROR}\n\nInstall the missing packages, then restart:\n"
                       f"pip install opencv-python numba").pack(anchor=tk.W, pady=(self.px(10), 0))


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class App:
    def __init__(self):
        enable_hidpi()
        self.root = tk.Tk()
        self.root.title("Image Processing - Assignment 3")
        self.theme = Theme(self.root)
        self._place_window()
        dark_title_bar(self.root)
        px = self.theme.px

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.current = None

        # Header: title on the left, page switcher on the right
        header = ttk.Frame(self.root, style="App.TFrame", padding=(px(20), px(16), px(20), px(2)))
        header.grid(row=0, column=0, sticky="ew")
        tk.Frame(header, bg=ACCENT, width=px(4)).pack(side=tk.LEFT, fill=tk.Y, padx=(0, px(12)))
        titles = ttk.Frame(header, style="App.TFrame")
        titles.pack(side=tk.LEFT)
        ttk.Label(titles, text="Image Processing", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(titles, text="Assignment 3  ·  Frequency-domain filtering & noise restoration",
                  style="Subtitle.TLabel").pack(anchor=tk.W)

        self.page = tk.StringVar(value="frequency")
        nav, _ = self.segmented(header, self.page,
                                [("frequency", "Frequency filtering"), ("noise", "Noise & filters")],
                                frame_style="Nav.TFrame", button_style="Nav.Toolbutton",
                                command=lambda: self.show(self.page.get()))
        nav.pack(side=tk.RIGHT)

        # Status bar (created before the pages, which report into it)
        footer = ttk.Frame(self.root, style="App.TFrame", padding=(px(22), 0, px(22), px(10)))
        footer.grid(row=2, column=0, sticky="ew")
        self.status = ttk.Label(footer, style="Status.TLabel")
        self.status.pack(side=tk.LEFT)
        ttk.Label(footer, style="Status.TLabel",
                  text="Ctrl+O  load image      Ctrl+S  save figure      Enter  apply").pack(side=tk.RIGHT)

        # Pages, stacked in the same cell
        content = ttk.Frame(self.root, style="App.TFrame", padding=(px(20), px(14)))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.pages = {
            "frequency": FrequencyPage(self, content),
            "noise": NoisePage(self, content) if exp2 else UnavailablePage(self, content),
        }
        for page in self.pages.values():
            page.frame.grid(row=0, column=0, sticky="nsew")

        self.root.bind("<Control-o>", lambda e: self.current.load_image())
        self.root.bind("<Control-s>", lambda e: self.current.save_figure())
        self.root.bind("<Return>", lambda e: self.current.apply())
        self.show("frequency")

    def _place_window(self):
        px = self.theme.px
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = min(px(1440), int(sw * 0.92)), min(px(880), int(sh * 0.86))
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 3)}")
        self.root.minsize(min(px(1040), w), min(px(660), h))

    def segmented(self, parent, var, options, frame_style="Seg.TFrame",
                  button_style="Seg.Toolbutton", command=None):
        """A row of radio buttons that look like one segmented control."""
        px = self.theme.px
        bar = ttk.Frame(parent, style=frame_style, padding=px(3))
        buttons = []
        for i, (value, text) in enumerate(options):
            button = ttk.Radiobutton(bar, text=text, value=value, variable=var,
                                     style=button_style, command=command, takefocus=False)
            button.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else px(3), 0))
            bar.columnconfigure(i, weight=1, uniform="segments")
            buttons.append(button)
        return bar, buttons

    def show(self, name):
        self.page.set(name)
        self.current = self.pages[name]
        self.current.frame.tkraise()
        self.status.config(text=self.current.status)

    def run(self):
        self.root.mainloop()


def main():
    App().run()


if __name__ == "__main__":
    main()
