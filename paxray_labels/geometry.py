import cc3d
import numpy as np
from scipy import ndimage as ndi

from paxray_labels.common import CATEGORY

VIEWS = ("frontal", "lateral")
_CONN2 = {4: None, 8: np.ones((3, 3))}


def rows(n):
    """source index of each of the 512 output rows / columns."""
    return np.minimum(np.floor(np.arange(512) * n / 512).astype(int), n - 1)


def rs(a):
    """2D array -> 512 x 512."""
    return a[rows(a.shape[0])][:, rows(a.shape[1])]


def proj(m, view):
    return rs(m.any(1)) if view == "frontal" else rs(m.any(2))[:, ::-1]


def lab(seg, *names):
    """union of categories given by name or id."""
    ids = [CATEGORY[n] if isinstance(n, str) else n for n in names]
    if len(ids) > 4:
        return np.isin(seg, ids)
    m = seg == ids[0]
    for i in ids[1:]:
        m |= seg == i
    return m


def bbox(m, pad=0):
    """bounding box of the 3D mask m as slices, widened by pad voxels per side; None if m is empty."""
    z = np.flatnonzero(m.any((1, 2)))
    if not len(z):
        return None
    sl = m[z[0]: z[-1] + 1]
    y, x = np.flatnonzero(sl.any((0, 2))), np.flatnonzero(sl.any((0, 1)))
    return tuple(slice(max(a[0] - pad, 0), min(a[-1] + 1 + pad, n)) for a, n in zip((z, y, x), m.shape))


def on_box(f, m, pad=0):
    """f applied to the bounding box of m (widened by pad), pasted into a zero array of m's shape."""
    o = np.zeros(m.shape, bool)
    sl = bbox(m, pad)
    if sl is not None:
        o[sl] = f(np.ascontiguousarray(m[sl]))
    return o


def lcc2(m, conn):
    """largest 4- or 8-connected component (2D)."""
    lb, n = ndi.label(m, _CONN2[conn])
    if n < 2:
        return m.copy()
    return lb == np.argmax(np.bincount(lb.ravel())[1:]) + 1


def lcc3(m, conn=26):
    """largest 3D component."""
    return on_box(lambda c: cc3d.largest_k(c, 1, connectivity=conn) > 0, m, 1) if m.any() else m


def _round(x):
    return int(np.floor(x + 0.5))


def n_voxels(ml, spacing):
    """number of voxels of a volume in ml (at least 1)."""
    return max(_round(ml * 1000.0 / np.prod(spacing)), 1)


def n_pixels(mm2, spacing):
    """number of pixels of an axial section area in mm^2 (at least 1)."""
    return max(_round(mm2 / (spacing[1] * spacing[2])), 1)


def n_slices(mm, spacing):
    """number of slices of a length in mm along z (at least 1)."""
    return max(_round(mm / spacing[0]), 1)


def kernel(mm, spacing):
    """odd kernel size per axis (z, y, x) of a centred kernel with extent mm (z, y, x)."""
    return tuple(max(2 * _round((e / s - 1) / 2) + 1, 1) for e, s in zip(mm, spacing))


def cross(size):
    """3D cross structuring element with the given odd size per axis."""
    s = np.zeros(size, bool)
    c = [k // 2 for k in size]
    s[:, c[1], c[2]] = s[c[0], :, c[2]] = s[c[0], c[1], :] = True
    return s


# 3D median filter extent (z, y, x)
MEDIAN_MM = (7.5, 2.07, 2.07)


def med3(m, spacing, size_mm=MEDIAN_MM):
    """3D binary median filter with a box of size_mm (z, y, x)."""
    if not m.any():
        return m
    k = kernel(size_mm, spacing)

    def _f(c):
        s = c.astype(np.uint16)
        for ax in range(3):
            s = ndi.correlate1d(s, np.ones(k[ax]), ax, mode="reflect")
        return s > int(np.prod(k)) // 2
    return on_box(_f, m, max(k))
