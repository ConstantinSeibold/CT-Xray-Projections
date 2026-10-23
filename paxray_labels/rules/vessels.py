import cc3d
import numpy as np
from scipy import ndimage as ndi

from paxray_labels.common import CATEGORY
from paxray_labels.geometry import VIEWS, bbox, lcc2, lcc3, med3, n_slices, proj

ABDOMEN = ["stomach", "small bowel", "duodenum", "liver", "pancreas", "kidney left", "kidney right", "esophagus"]
NO_LATERAL = {"kidney left", "kidney right"}
SPLIT_RUN_MM, ARCH_OFF_MM = 30.0, 2.0
TRACHEA_CUT_MM, BIF_START_MM, BIF_LENGTH_MM = 4.0, 25.0, 42.5
AORTA_START_MM = 12.5


def aorta_split_level(AL, spacing):
    sz = spacing[0]
    zz = np.nonzero(AL.any((1, 2)))[0]
    _, ys, xs = bbox(AL)
    n = np.zeros(AL.shape[0], int)
    n[zz[0]: zz[-1] + 1] = [ndi.label(AL[z, ys, xs])[1] for z in range(zz[0], zz[-1] + 1)]
    run = max(int(np.ceil(SPLIT_RUN_MM / sz)), 2)
    zs = None
    for z in range(zz[0] + n_slices(AORTA_START_MM, spacing), zz[-1] + 1):
        if (n[z:z + run] >= 2).all() and z + run <= zz[-1] + 1:
            zs = z
            break
    if zs is None:
        zs = (zz[0] + zz[-1]) // 2
    return zs + int(round(ARCH_OFF_MM / sz))


def aorta_parts(AL, z0):
    """(arch, ascending, descending)."""
    arch = AL.copy(); arch[z0:] = 0
    B = AL.copy(); B[:z0] = 0
    sl = bbox(B, 1)
    if sl is None:
        return arch, np.zeros_like(B), np.zeros_like(B)
    Bc = np.ascontiguousarray(B[sl])
    lb = cc3d.connected_components(Bc, connectivity=26)
    s = np.bincount(lb.ravel()); s[0] = 0
    o = np.argsort(s)[::-1][:2]
    c1 = lb == o[0]
    c2 = lb == o[1] if len(o) > 1 and s[o[1]] > 0 else np.zeros_like(Bc)
    rest = Bc & ~c1 & ~c2
    if c2.any() and rest.any():
        d1 = ndi.distance_transform_edt(~c1); d2 = ndi.distance_transform_edt(~c2)
        c1, c2 = c1 | (rest & (d1 <= d2)), c2 | (rest & (d2 < d1))
    C1, C2 = np.zeros_like(B), np.zeros_like(B)
    C1[sl], C2[sl] = c1, c2
    if not c2.any():
        return arch, C1, C2
    y1 = (np.nonzero(c1)[1] + sl[1].start).mean(); y2 = (np.nonzero(c2)[1] + sl[1].start).mean()
    return (arch, C1, C2) if y1 < y2 else (arch, C2, C1)


def trachea_cut_slice(T, E, mm, spacing):
    """first slice where the trachea is within mm of the esophagus."""
    for z in np.nonzero(T.any((1, 2)))[0]:
        if E[z].any():
            d = ndi.distance_transform_edt(~E[z], sampling=spacing[1:])[T[z]].min()
            if d <= mm:
                return int(z)
    return 0


def bifurcation_split(T, start):
    tz = np.nonzero(T.any((1, 2)))[0]
    for z in range(tz[0] + start, tz[-1] + 1):
        if ndi.label(T[z])[1] >= 2:
            return z
    return tz[-1]


def derive(case, raw_abdomen=()):
    """raw_abdomen: (view, class) pairs whose abdominal organ is taken from the raw labels."""
    seg = case.pp
    segs = {"pp": seg, "raw": case.raw}
    out = {"frontal": {}, "lateral": {}}

    for n in ABDOMEN:
        m3 = {}
        for v in VIEWS:
            if v == "lateral" and n in NO_LATERAL:
                continue
            w = "raw" if (v, n) in raw_abdomen else "pp"
            if w not in m3:
                m3[w] = med3(segs[w] == CATEGORY[n], case.spacing)
            out[v][n] = ndi.binary_fill_holes(lcc2(proj(m3[w], v), 8))

    AL = lcc3(seg == CATEGORY["aorta"])
    arch, asc, desc = aorta_parts(AL, aorta_split_level(AL, case.spacing)) if AL.any() else (AL, AL, AL)
    vol = {"pulmonary artery": seg == CATEGORY["pulmonary artery"], "inferior vena cava": seg == CATEGORY["inferior vena cava"],
           "aorta": AL, "aortic arch": arch, "ascending aorta": asc, "descending aorta": desc}

    T = seg == CATEGORY["trachea"]
    tr = T.copy()
    tr[:trachea_cut_slice(T, seg == CATEGORY["esophagus"], TRACHEA_CUT_MM, case.spacing)] = 0
    vol["trachea"] = tr
    bif = T.copy()
    if T.any():
        bif[:max(bifurcation_split(T, n_slices(BIF_START_MM, case.spacing)) - n_slices(BIF_LENGTH_MM, case.spacing), 0)] = 0
    vol["tracheal bifurcation"] = bif

    for k, m in vol.items():
        for v in VIEWS:
            out[v][k] = proj(m, v)
    return out
