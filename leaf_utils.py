"""Shared helpers for the Leaffliction programs.

Dataset discovery, image augmentation, PlantCV based image
transformations and a few small display helpers.
"""
import io
import os
import re
import sys
from collections import OrderedDict

import cv2
import numpy as np

IMG_EXT = ('.jpg', '.jpeg', '.png')
AUG_TYPES = ('Flip', 'Rotate', 'Skew', 'Shear', 'Crop', 'Distortion')
AUG_RE = re.compile(
    r'_(' + '|'.join(AUG_TYPES) + r')\d*\.[A-Za-z]+$')


class LeafError(Exception):
    """Error reported to the user without a traceback."""


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------

def is_image(name):
    return name.lower().endswith(IMG_EXT)


def is_augmented(name):
    """True for files created by Augmentation (name_Flip.JPG, ...)."""
    return AUG_RE.search(name) is not None


def read_image(path):
    """Read an image (BGR) or raise LeafError."""
    img = None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError as err:
        raise LeafError('cannot read %s: %s' % (path, err.strerror))
    if img is None:
        raise LeafError('%s is not a valid image' % path)
    return img


def write_image(path, img):
    """Write an image; the extension of `path` selects the format."""
    ext = os.path.splitext(path)[1] or '.jpg'
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext.lower() in (
        '.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise LeafError('cannot encode %s' % path)
    with open(path, 'wb') as fd:
        fd.write(buf.tobytes())


def check_dir(path):
    if not os.path.isdir(path):
        raise LeafError('%s is not a directory' % path)


def class_dirs(root, skip_augmented=False):
    """Return {directory: [image paths]} for every directory under root
    that directly contains images."""
    found = {}
    for cur, dirs, files in os.walk(root):
        dirs.sort()
        imgs = sorted(f for f in files if is_image(f))
        if skip_augmented:
            imgs = [f for f in imgs if not is_augmented(f)]
        if imgs:
            found[cur] = [os.path.join(cur, f) for f in imgs]
    return found


def group_by_plant(root):
    """Group class directories by plant type (their parent directory).

    Returns an ordered list of (plant name, {class name: count})."""
    groups = OrderedDict()
    root = os.path.normpath(root)
    for cur, imgs in class_dirs(root).items():
        parent = os.path.dirname(cur)
        if cur == root:
            parent = os.path.dirname(os.path.abspath(root))
        plant = os.path.basename(os.path.abspath(parent))
        groups.setdefault(parent, (plant, OrderedDict()))
        groups[parent][1][os.path.basename(cur)] = len(imgs)
    return list(groups.values())


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def _flip(img, rng):
    return cv2.flip(img, int(rng.integers(0, 2)))


def _warp_affine(img, mat):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def _rotate(img, rng):
    h, w = img.shape[:2]
    angle = float(rng.uniform(15, 45)) * rng.choice((-1, 1))
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return _warp_affine(img, mat)


def _skew(img, rng):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jit = rng.uniform(0.05, 0.18, size=(4, 2)) * rng.choice(
        (-1, 1), size=(4, 2)) * np.float32([w, h])
    dst = (src + jit).astype(np.float32)
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, mat, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)


def _shear(img, rng):
    h, w = img.shape[:2]
    s = float(rng.uniform(0.15, 0.35)) * rng.choice((-1, 1))
    if rng.integers(0, 2):
        mat = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
    else:
        mat = np.float32([[1, 0, 0], [s, 1, -s * w / 2]])
    return _warp_affine(img, mat)


def _crop(img, rng):
    h, w = img.shape[:2]
    frac = float(rng.uniform(0.65, 0.85))
    ch, cw = int(h * frac), int(w * frac)
    y = int(rng.integers(0, h - ch + 1))
    x = int(rng.integers(0, w - cw + 1))
    return cv2.resize(img[y:y + ch, x:x + cw], (w, h),
                      interpolation=cv2.INTER_CUBIC)


def _distortion(img, rng):
    h, w = img.shape[:2]
    size = max(h, w)
    alpha, sigma = size * 0.12, size * 0.04
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha / (sigma * 0.1 + 1e-6)
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha / (sigma * 0.1 + 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + dx, gy + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


AUGMENTATIONS = OrderedDict([
    ('Flip', _flip), ('Rotate', _rotate), ('Skew', _skew),
    ('Shear', _shear), ('Crop', _crop), ('Distortion', _distortion),
])


def augment(img, kind, rng):
    return AUGMENTATIONS[kind](img, rng)


def free_name(directory, stem, kind, ext):
    """name_Kind.ext, or name_Kind2.ext... when already taken."""
    path = os.path.join(directory, '%s_%s%s' % (stem, kind, ext))
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(directory, '%s_%s%d%s' % (stem, kind, n, ext))
    return path


# --------------------------------------------------------------------------
# Image transformations (PlantCV)
# --------------------------------------------------------------------------

def _pcv():
    """Import PlantCV lazily with debug output disabled."""
    from plantcv import plantcv as pcv
    pcv.params.debug = None
    pcv.params.verbose = False
    return pcv


def leaf_mask(img):
    """Binary mask (255 = leaf) from the Lab green-magenta channel."""
    pcv = _pcv()
    chan = pcv.rgb2gray_lab(rgb_img=img, channel='a')
    blur = pcv.gaussian_blur(img=chan, ksize=(5, 5))
    mask = pcv.threshold.otsu(gray_img=blur, object_type='dark')
    mask = pcv.fill(bin_img=mask, size=200)
    mask = pcv.fill_holes(bin_img=mask)
    if cv2.countNonZero(mask) < 0.02 * mask.size:
        # segmentation failed (blank picture, ...): keep everything
        mask = np.full(mask.shape, 255, dtype=np.uint8)
    return mask


def apply_mask(img, mask, color=(255, 255, 255)):
    out = np.empty_like(img)
    out[:] = color
    out[mask > 0] = img[mask > 0]
    return out


def _histogram_channels(img, mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return OrderedDict([
        ('blue', (img[:, :, 0], 256, (0, 0.9, 0.9))),
        ('blue-yellow', (lab[:, :, 2], 256, (0.9, 0.8, 0.0))),
        ('green', (img[:, :, 1], 256, (0.1, 0.7, 0.1))),
        ('green-magenta', (lab[:, :, 1], 256, (0.8, 0.0, 0.8))),
        ('hue', (hsv[:, :, 0], 180, (0.5, 0.3, 0.9))),
        ('lightness', (lab[:, :, 0], 256, (0.5, 0.5, 0.5))),
        ('red', (img[:, :, 2], 256, (0.9, 0.1, 0.1))),
        ('saturation', (hsv[:, :, 1], 256, (0.0, 0.7, 0.7))),
        ('value', (hsv[:, :, 2], 256, (0.9, 0.6, 0.1))),
    ])


def histogram_figure(img, mask):
    """Colour histogram of the leaf pixels as a matplotlib Figure."""
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.subplots()
    sel = mask > 0
    for name, (chan, bins, color) in _histogram_channels(img, mask).items():
        hist = np.bincount(chan[sel].ravel(), minlength=bins)[:bins]
        hist = hist * 100.0 / max(int(sel.sum()), 1)
        ax.plot(np.arange(bins), hist, color=color, label=name)
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Proportion of pixels (%)')
    ax.set_title('Color histogram')
    ax.legend(title='Color Channel', fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figure_to_bgr(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    buf = io.BytesIO()
    canvas.print_figure(buf, format='png')
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


TRANSFORM_NAMES = ('Original', 'Gaussian_blur', 'Mask', 'Roi_objects',
                   'Analyze_object', 'Pseudolandmarks', 'Color_histogram')


def transformations(img):
    """Apply every transformation, return an ordered {name: BGR image}.

    The binary mask is returned under the key '_mask' as well."""
    pcv = _pcv()
    out = OrderedDict()
    out['Original'] = img
    out['Gaussian_blur'] = pcv.gaussian_blur(img=img, ksize=(9, 9))

    mask = leaf_mask(img)
    out['Mask'] = apply_mask(img, mask)

    # Regions of interest: keep the objects overlapping the frame
    h, w = img.shape[:2]
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=h, w=w)
    kept = pcv.roi.filter(mask=mask, roi=roi, roi_type='partial')
    contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    roi_img = img.copy()
    roi_img[kept > 0] = (0.5 * roi_img[kept > 0]
                         + 0.5 * np.array((0, 255, 0))).astype(np.uint8)
    cv2.drawContours(roi_img, contours, -1, (0, 200, 0), 2)
    cv2.rectangle(roi_img, (1, 1), (w - 2, h - 2), (255, 0, 0), 3)
    out['Roi_objects'] = roi_img

    # Shape analysis of the leaf
    labeled, nlab = pcv.create_labels(mask=kept)
    if nlab > 0:
        shape = pcv.analyze.size(img=img, labeled_mask=labeled,
                                 n_labels=nlab)
        pcv.outputs.clear()
    else:
        shape = img.copy()
    out['Analyze_object'] = shape

    # Pseudolandmarks along the x axis
    pts = img.copy()
    try:
        top, bot, cen = pcv.homology.x_axis_pseudolandmarks(
            img=img, mask=kept)
        pcv.outputs.clear()
        for arr, col in ((top, (255, 0, 0)), (bot, (0, 0, 255)),
                         (cen, (0, 200, 0))):
            for p in np.asarray(arr).reshape(-1, 2):
                cv2.circle(pts, (int(p[0]), int(p[1])), 4, col, -1)
    except Exception:  # noqa: BLE001 - PlantCV raises on degenerate masks
        pass
    out['Pseudolandmarks'] = pts

    fig = histogram_figure(img, kept)
    out['Color_histogram'] = figure_to_bgr(fig)
    out['_mask'] = kept
    return out


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

def can_display():
    """True when matplotlib can open a window."""
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in ('agg', 'pdf', 'svg', 'ps', 'cairo', 'template'):
        return False
    if sys.platform.startswith('linux'):
        return bool(os.environ.get('DISPLAY')
                    or os.environ.get('WAYLAND_DISPLAY'))
    return True


def show_images(items, title=None, cols=3, block=True):
    """Show [(name, BGR image)] in a grid; no-op without a display."""
    import matplotlib.pyplot as plt
    if not can_display():
        print('(no display available: images not shown)', file=sys.stderr)
        return
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (name, im) in zip(axes, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if block:
        plt.show()
