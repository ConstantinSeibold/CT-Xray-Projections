import cc3d
import numpy as np
from scipy import ndimage as ndi

from paxray_labels.common import CATEGORY
from paxray_labels.geometry import (VIEWS, bbox, cross, kernel, lab, lcc2, med3, n_pixels, n_slices, on_box, proj,
                                    rows)

CHAMBERS = ["heart atrium left", "heart atrium right", "heart myocardium", "heart ventricle left", "heart ventricle right"]
HEART_IDS = ["heart", "heart atrium left", "heart atrium right", "heart myocardium", "heart ventricle left", "heart ventricle right"]
HEART_ALL = HEART_IDS + ["heart tissue"]
CM_IDS = HEART_ALL + ["aorta", "pulmonary artery", "mediastinal tissue", "bronchi"]
CROSS = ndi.generate_binary_structure(2, 1)
# extents (z, y, x) in mm of the majority box and of the opening cross; lengths along z in mm; areas in mm^2
MAJORITY_MM = (12.5, 3.46, 3.46)
OPENING_MM = (7.5, 2.07, 2.07)
SPLIT_RUN_MM, SPLIT_OFF_MM = 7.5, 5.0
MT_MIN_MM2, SPLIT_MIN_MM2 = 9.56, 47.8


def lcc3(m, conn=26):
    """largest 3D component."""
    def _l(c):
        l = cc3d.connected_components(c, connectivity=conn)
        return c if l.max() < 2 else l == np.argmax(np.bincount(l.ravel())[1:]) + 1
    return on_box(_l, m, 1) if m.any() else m


def zb_mt(seg, spacing):
    """bottom slice of the mediastinal tissue; -1 if absent."""
    T = n_pixels(MT_MIN_MM2, spacing)
    a = np.count_nonzero(seg == CATEGORY["mediastinal tissue"], axis=(1, 2))
    if not a.any():
        return -1
    z = int(np.argmax(a))
    while z + 1 < len(a) and a[z + 1] >= T:
        z += 1
    return z


def cm_majority(seg, spacing):
    """box mean of the cardiomediastinum labels down to the bottom of the mediastinal tissue."""
    z = zb_mt(seg, spacing)
    k = kernel(MAJORITY_MM, spacing)
    u = lab(seg[: z + 1], *CM_IDS)
    f = np.zeros(u.shape, np.float32)
    sl = bbox(u, max(k) // 2 + 1)
    if sl is not None:
        f[sl] = ndi.uniform_filter(u[sl].astype(np.float32), k)
    return f


def cm3d(seg, t, maj):
    out = np.zeros(seg.shape, bool)
    out[: len(maj)] = lcc3(maj > t)
    return out


def t4_bottom_med(seg, spacing):
    """bottom row of the frontal projection of the median-filtered T4 label."""
    m = med3(seg == CATEGORY["vertebrae T4"], spacing)
    if not m.any():
        m = seg == CATEGORY["vertebrae T4"]
    if not m.any():
        return None
    return int(np.nonzero(lcc2(proj(m, "frontal"), 4).any(1))[0].max())


def aorta_parts(seg, spacing):
    ao = seg == CATEGORY["aorta"]
    zs = np.nonzero(ao.any((1, 2)))[0]
    if not len(zs):
        return ao, ao
    _, yb, xb = bbox(ao)
    min_px = n_pixels(SPLIT_MIN_MM2, spacing)

    def _split(z):
        sec = np.ascontiguousarray(ao[z, yb, xb])
        cnt = np.sort(np.bincount(cc3d.connected_components(sec, connectivity=8).ravel())[1:])[::-1]
        return len(cnt) >= 2 and cnt[1] >= min_px and cnt[1] >= 0.2 * cnt[0]
    sp = {z: _split(z) for z in zs}
    z2 = next((z for z in zs if all(sp.get(z + i, False) for i in range(n_slices(SPLIT_RUN_MM, spacing)))), zs.max())
    z0 = z2 + n_slices(SPLIT_OFF_MM, spacing)
    arch = ao.copy(); arch[z0:] = 0
    below = ao.copy(); below[:z0] = 0
    asc = np.zeros_like(ao)
    sl = bbox(below, 1)
    if sl is not None:
        l = cc3d.connected_components(np.ascontiguousarray(below[sl]), connectivity=26)
        cnt = np.bincount(l.ravel())[1:]
        big = np.argsort(cnt)[::-1][:2] + 1
        ys = [(np.nonzero(l == b)[1] + sl[1].start).mean() for b in big]
        asc[sl] = l == big[int(np.argmin(ys))]
    return arch, asc


def _rowrange(m):
    any_ = m.any(1); W = m.shape[1]
    mn = np.where(any_, np.argmax(m, 1), -1)
    mx = np.where(any_, W - 1 - np.argmax(m[:, ::-1], 1), -1)
    return any_, mn, mx


def amp(LO, heart, asc, arch):
    cols = np.arange(LO.shape[1])[None, :]
    C0 = np.nonzero(asc.any(0))[0].min() - 1 if asc.any() else LO.shape[1]
    mid = heart | asc | (arch & (cols > C0))
    a, mn, mx = _rowrange(mid)
    mn = np.where(a, mn, 10 ** 6)[:, None]; mx = np.where(a, mx, -1)[:, None]
    P = LO & (cols <= np.minimum(C0, mn - 1))
    M = LO & (cols >= mn) & (cols <= mx)
    A = (LO & (cols > mx) & a[:, None]) | (LO & (cols > C0) & ~a[:, None])
    return A, M, P


# (class, view): (3D labels, 3D filter)
CHAMBER_SPEC = {
    ("heart atrium left", "frontal"): ("pp", "med3"), ("heart atrium left", "lateral"): ("pp", "med3"),
    ("heart atrium right", "frontal"): ("raw", "none"), ("heart atrium right", "lateral"): ("raw", "none"),
    ("heart myocardium", "frontal"): ("pp", "med3"), ("heart myocardium", "lateral"): ("pp", "med3"),
    ("heart ventricle left", "frontal"): ("pp", "med3"), ("heart ventricle left", "lateral"): ("pp", "lcc"),
    ("heart ventricle right", "frontal"): ("pp", "med3"), ("heart ventricle right", "lateral"): ("pp", "med3"),
}


def _filt(m, f, spacing):
    if f == "med3":
        return lcc3(med3(m, spacing))
    if f == "lcc":
        return lcc3(m)
    return m


def derive(case):
    segs = {"pp": case.pp, "raw": case.raw}
    seg = segs["pp"]
    sp = case.spacing
    maj = cm_majority(seg, sp)
    cm = cm3d(seg, 0.6, maj)
    cm_up = cm3d(seg, 0.55, maj)
    t4b = t4_bottom_med(seg, sp)
    if t4b is None:
        cm_up[:] = 0
    else:
        cm_up[rows(seg.shape[0])[t4b] + 1:] = 0
    arch3, asc3 = aorta_parts(seg, sp)
    heart3 = _filt(lab(seg, *HEART_IDS), "med3", sp)
    heart_all = lab(seg, *HEART_ALL)
    ch3 = {}
    for (n, v), (w, f) in CHAMBER_SPEC.items():
        if (n, w, f) not in ch3:
            ch3[(n, w, f)] = _filt(segs[w] == CATEGORY[n], f, sp)
    opening = cross(kernel(OPENING_MM, sp))
    out = {}
    for view in VIEWS:
        d = {}
        C = ndi.binary_fill_holes(proj(cm, view))
        u = on_box(lambda c: ndi.binary_opening(c, opening), cm_up, max(opening.shape) // 2 + 1) if view == "lateral" else cm_up
        if t4b is None:
            up, lo = np.zeros_like(C), np.zeros_like(C)
        else:
            up = ndi.binary_fill_holes(proj(lcc3(u), view)); up[t4b + 1:] = 0
            lo = C.copy(); lo[: t4b + 1] = 0
        d["cardiomediastinum"], d["upper mediastinum"], d["lower mediastinum"] = C, up, lo
        d["heart"] = ndi.binary_fill_holes(lcc2(proj(heart3, view), 4))
        for n in CHAMBERS:
            w, f = CHAMBER_SPEC[(n, view)]
            d[n] = ndi.binary_fill_holes(lcc2(proj(ch3[(n, w, f)], view), 4))
        if view == "lateral" and arch3.any() and heart_all.any():
            A, M, Pm = amp(lo, proj(heart_all, view), proj(asc3, view), proj(arch3, view))
            d["anterior mediastinum"] = lcc2(ndi.binary_opening(A, CROSS), 4)
            d["middle mediastinum"], d["posterior mediastinum"] = M, lcc2(Pm, 4)
        out[view] = d
    return out
