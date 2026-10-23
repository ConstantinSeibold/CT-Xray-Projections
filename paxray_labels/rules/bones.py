import functools

import cc3d
import numpy as np
from scipy import ndimage as ndi

from paxray_labels.common import CATEGORY
from paxray_labels.geometry import VIEWS, bbox, lab, lcc2, lcc3, n_voxels, proj

V = [f"vertebrae {p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 5)) for i in range(1, n + 1)]
AGG = {"spine": V, "cervical spine": V[:7], "thoracic spine": V[7:19], "lumbar spine": V[19:]}
RAW_MIN_ML = 0.012


def _zc(m):
    return float(np.median(np.flatnonzero(m.any((1, 2)))))


def order_keep(ms):
    """indices of the cranio-caudally ordered subset of maximum total voxel count."""
    items = [(i, _zc(m), int(np.count_nonzero(m))) for i, m in enumerate(ms) if m.any()]
    n = len(items)
    if not n:
        return set()
    best, prev = [it[2] for it in items], [-1] * n
    for a in range(n):
        for b in range(a):
            if items[b][1] < items[a][1] and best[b] + items[a][2] > best[a]:
                best[a], prev[a] = best[b] + items[a][2], b
    a, keep = int(np.argmax(best)), set()
    while a >= 0:
        keep.add(items[a][0])
        a = prev[a]
    return keep


def vertebrae3d(pp, raw, spacing):
    raw_min = n_voxels(RAW_MIN_ML, spacing)
    rl = [lcc3(raw == CATEGORY[v]) for v in V]
    ms = [(pp == CATEGORY[v]) if np.count_nonzero(rl[i]) >= raw_min else np.zeros(pp.shape, bool) for i, v in enumerate(V)]
    keep = order_keep(ms)
    ms = [m if i in keep else np.zeros_like(m) for i, m in enumerate(ms)]
    # missing vertebrae: largest raw component if it lies between its neighbours
    for i in range(len(V)):
        if ms[i].any() or np.count_nonzero(rl[i]) < raw_min:
            continue
        z = _zc(rl[i])
        above = [_zc(ms[j]) for j in range(i) if ms[j].any()]
        below = [_zc(ms[j]) for j in range(i + 1, len(V)) if ms[j].any()]
        if (not above or max(above) < z) and (not below or min(below) > z):
            ms[i] = rl[i]
    return ms


def sternum3d(seg):
    """sternum labels plus the enclosed unlabeled components (6-connected) touching them."""
    base = lab(seg, "sternum", "manubrium")
    for pad in (2, 16, 64, max(seg.shape)):
        sl = bbox(base, pad)
        if sl is None:
            return base
        z0 = seg[sl] == 0
        lb = cc3d.connected_components(z0.astype(np.uint8), connectivity=6)
        k = np.unique(lb[ndi.binary_dilation(base[sl]) & z0])
        outer, inner = set(), set()
        for ax, (s, n) in enumerate(zip(sl, seg.shape)):
            for face, on_border in ((0, s.start == 0), (-1, s.stop == n)):
                (outer if on_border else inner).update(np.unique(lb.take(face, ax)).tolist())
        k = [i for i in k.tolist() if i > 0 and i not in outer]
        if not any(i in inner for i in k):
            out = base.copy()
            out[sl] |= np.isin(lb, k)
            return out


def derive(case):
    raw, pp = case.raw, case.pp
    out = {"frontal": {}, "lateral": {}}
    ms = vertebrae3d(pp, raw, case.spacing)
    st = sternum3d(pp)
    cart = raw == CATEGORY["rib cartilage"]
    side = {f"{pre} {s}": lcc3(pp == CATEGORY[f"{pre} {s}"], 6) for pre in ("clavicle", "scapula") for s in ("left", "right")}
    for view in VIEWS:
        d = out[view]
        pv = {v: proj(m, view) for v, m in zip(V, ms)}
        for v in V:
            d[v] = lcc2(pv[v], 8)
        for a, vs in AGG.items():
            d[a] = functools.reduce(np.logical_or, [pv[v] for v in vs])
        d["sternum"] = lcc2(proj(st, view), 8)
        d["rib_cartilage"] = ndi.binary_fill_holes(proj(cart, view))
        if view == "frontal":
            for n, m in side.items():
                d[n] = proj(m, view)
            d["clavicles"] = d["clavicle left"] | d["clavicle right"]
            d["scapulas"] = d["scapula left"] | d["scapula right"]
        else:
            d["clavicles"] = lcc2(proj(lab(raw, "clavicle left", "clavicle right"), view), 4)
            d["scapulas"] = proj(side["scapula left"] | side["scapula right"], view)
    return out
