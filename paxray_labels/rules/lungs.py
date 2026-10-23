import functools

import cc3d
import numpy as np
from scipy import ndimage as ndi

from paxray_labels.geometry import VIEWS, bbox, lab, lcc2, lcc3, med3, n_voxels, proj, rows

R3 = ["lung upper lobe right", "lung middle lobe right", "lung lower lobe right"]
L2 = ["lung upper lobe left", "lung lower lobe left"]
MIRROR = {"lung upper lobe right": "lung upper lobe left", "lung upper lobe left": "lung upper lobe right",
          "lung lower lobe right": "lung lower lobe left", "lung lower lobe left": "lung lower lobe right"}
ZONES = (("upper zone lung", "upper"), ("mid zone lung", "mid"), ("lung base", "base"))


def zext(m3, spacing, min_ml=0.06):
    """(top, bottom) slice of m3 after dropping 3D components (26-conn) smaller than min_ml."""
    min_size = n_voxels(min_ml, spacing)
    sl = bbox(m3, 1)
    if sl is None:
        return None
    c = np.ascontiguousarray(m3[sl])
    lb, N = cc3d.connected_components(c, connectivity=26, return_N=True)
    sz = np.bincount(lb.ravel())
    keep = np.isin(lb, np.flatnonzero(sz >= min_size)[1:]) if N else c
    z = np.flatnonzero((keep if keep.any() else c).any((1, 2)))
    return int(z[0]) + sl[0].start, int(z[-1]) + sl[0].start


def bands(e, H):
    if e is None:
        z = np.zeros(H, bool)
        return {"upper": z, "mid": z, "base": z}
    zt, zb = e
    d = zb - zt
    u = zb - np.arange(H)
    return {"base": u < int(d / 3), "mid": (u >= int(d / 3)) & (u < int(2 * d / 3)), "upper": u >= int(2 * d / 3)}


def midline_x(seg):
    m = lab(seg, "sternum", *[f"vertebrae T{k}" for k in range(1, 13)])
    sl = bbox(m)
    return (np.nonzero(m[sl])[2] + sl[2].start).mean() if sl is not None else seg.shape[2] / 2


def _labels_x0(m, connectivity):
    """3D components of m on its bounding box, extended to x = 0; returns (labels, N, slices)."""
    sl = bbox(m, 1)
    if sl is None:
        return None, 0, None
    sl = sl[:2] + (slice(0, sl[2].stop),)
    lb, N = cc3d.connected_components(np.ascontiguousarray(m[sl]), connectivity=connectivity, return_N=True)
    return lb, N, sl


def lobes3d(seg, mx, spacing, min_ml=1.195):
    min_size = n_voxels(min_ml, spacing)
    """lobes; components of an upper / lower lobe label on the other side of the midline go to the mirrored lobe."""
    masks = {n: lab(seg, n) for n in R3 + L2}
    moved = {n: np.zeros(seg.shape, bool) for n in R3 + L2}
    for n, tgt in MIRROR.items():
        lb, N, sl = _labels_x0(masks[n], 26)
        if not N:
            continue
        st = cc3d.statistics(lb)
        cen, sz = st["centroids"][1:, 2], st["voxel_counts"][1:]
        wrong = [i + 1 for i in range(N) if sz[i] >= min_size and ((cen[i] >= mx) if "right" in n else (cen[i] < mx))]
        if wrong:
            moved[tgt][sl] |= np.isin(lb, wrong)
    return {n: lcc3(masks[n] | moved[n]) for n in R3 + L2}


def breast3d(seg, spacing, mx, min_ml=0.5):
    """(right, left) 3D breast masks."""
    u = med3(lab(seg, "breast left", "breast right"), spacing)
    z = np.zeros_like(u)
    lb, N, sl = _labels_x0(u, 26)
    if not N:
        return z, z
    st = cc3d.statistics(lb)
    sz, cen = st["voxel_counts"], st["centroids"][:, 2]
    ml = np.prod(spacing) / 1000.0
    keep = [i for i in range(1, N + 1) if sz[i] * ml >= min_ml]
    R, L = np.zeros_like(u), np.zeros_like(u)
    R[sl], L[sl] = np.isin(lb, [i for i in keep if cen[i] < mx]), np.isin(lb, [i for i in keep if cen[i] >= mx])
    return R, L


def derive(case):
    sp, sr = case.pp, case.raw
    H = sp.shape[0]
    out = {"frontal": {}, "lateral": {}}
    F, Lt = out["frontal"], out["lateral"]

    # lobes and lungs
    mx = midline_x(sr)
    lobes = lobes3d(sr, mx, case.spacing)
    side = {"right": functools.reduce(np.logical_or, [lobes[n] for n in R3]),
            "left": functools.reduce(np.logical_or, [lobes[n] for n in L2])}
    lung = side["right"] | side["left"]
    for n in R3 + L2:
        F[n] = proj(lobes[n], "frontal")
    F["right lung"] = proj(side["right"], "frontal")
    F["left lung"] = proj(side["left"], "frontal")
    F["lung"] = proj(lung, "frontal")
    Lt["lung"] = proj(lung, "lateral")

    # zones: thirds of the lung's cranio-caudal extent
    raw_side = {"right": lab(sr, *R3), "left": lab(sr, *L2)}
    zr = rows(H)
    for s in ("right", "left"):
        b = bands(zext(raw_side[s], case.spacing), H)
        for zn, zk in ZONES:
            F[f"{s} {zn}"] = F[f"{s} lung"] & b[zk][zr][:, None]
    b = bands(zext(raw_side["right"] | raw_side["left"], case.spacing), H)
    for zn, zk in ZONES:
        for v in VIEWS:
            out[v][zn] = out[v]["lung"] & b[zk][zr][:, None]

    # apical zones: lung rows above the lower edge of the clavicles
    clav = proj(lab(sp, "clavicle left", "clavicle right"), "frontal")
    r = np.flatnonzero(clav.any(1))
    keep = (np.arange(512) <= (r[-1] if len(r) else -1))[:, None]
    F["apical zone lung"] = F["lung"] & keep
    Lt["apical zone lung"] = Lt["lung"] & keep
    F["right apical zone lung"] = F["right lung"] & keep
    F["left apical zone lung"] = F["left lung"] & keep

    # diaphragm
    dia3 = lab(sr, "abdominal tissue", "liver", "stomach", "small bowel")
    for v in VIEWS:
        d = proj(dia3, v)
        out[v]["diaphragm"] = ndi.binary_fill_holes(lcc2(d, 4)) if d.any() else d
    d = F["diaphragm"]
    xs = np.flatnonzero(d.any(0))
    sx = int(np.mean(xs)) if len(xs) else 256
    cols = np.arange(512)[None, :]
    F["right hemidiaphragm"] = d & (cols < sx)
    F["left hemidiaphragm"] = d & (cols >= sx)

    # breast
    R, Lf = breast3d(sr, case.spacing, mx)
    F["breast right"], F["breast left"] = proj(R, "frontal"), proj(Lf, "frontal")
    F["breast"] = F["breast right"] | F["breast left"]
    Lt["breast"] = lcc2(proj(R | Lf, "lateral"), 4)
    return out
