import cc3d
import numpy as np

from paxray_labels.categories import RIBS
from paxray_labels.geometry import bbox, lab, rs

SIDES = ("right", "left")


def ordn(k):
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


def _anterior(s, k, zz, yy, xx, sy):
    # anterior/posterior split at the most lateral voxel
    ex = xx if s == "left" else -xx
    if k == 1:
        return yy <= yy[ex >= ex.max() - 4.0 / sy].mean()
    sel = np.flatnonzero(ex == ex.max())
    j = sel[np.lexsort((yy[sel], zz[sel]))[-1]]  # ties: most inferior, then most posterior
    return yy <= yy[j]


def rib_parts(seg, sy):
    """{(side, k): (zz, yy, xx, anterior or None)} of the numbered 3D rib instances."""
    lb = cc3d.connected_components(lab(seg, *RIBS), connectivity=6)
    st = cc3d.statistics(lb)
    vox, cen, bbs = st["voxel_counts"], st["centroids"], st["bounding_boxes"]
    tv = lab(seg, *[f"vertebrae T{i}" for i in range(1, 13)])
    sl = bbox(tv)
    tx = (np.nonzero(tv[sl])[2] + sl[2].start).mean() if sl is not None else seg.shape[2] / 2
    out = {}
    for s in SIDES:
        ids = [i for i in range(1, len(vox)) if (cen[i][2] < tx) == (s == "right")]
        ids = sorted(ids, key=lambda i: -int(vox[i]))[:12]
        pts = {}
        for i in ids:
            sl = bbs[i]
            pts[i] = np.argwhere(lb[sl] == i) + [sl[0].start, sl[1].start, sl[2].start]
        # numbered from the top by the topmost voxel
        ids = sorted(ids, key=lambda i: pts[i][:, 0].min())
        for k, i in enumerate(ids, 1):
            zz, yy, xx = pts[i].T
            out[(s, k)] = (zz, yy, xx, None if k == 12 else _anterior(s, k, zz, yy, xx, sy))
    return out


def derive(case):
    raw = case.raw
    Z, Y, X = raw.shape
    parts = rib_parts(raw, case.spacing[1])
    E = np.zeros((512, 512), bool)
    res = {"frontal": {}, "lateral": {}}

    def fr(zz, xx):
        m = np.zeros((Z, X), bool); m[zz, xx] = True; return rs(m)

    def la(zz, yy):
        m = np.zeros((Z, Y), bool); m[zz, yy] = True; return rs(m)[:, ::-1]

    for k in range(1, 13):
        for ap in (("posterior",) if k == 12 else ("anterior", "posterior")):
            uf = E.copy(); ul = E.copy()
            for s in SIDES:
                nm = f"{ap} {ordn(k)} rib {s}"
                if (s, k) not in parts:
                    continue
                zz, yy, xx, a = parts[(s, k)]
                sel = np.ones(len(zz), bool) if a is None else (a if ap == "anterior" else ~a)
                if sel.any():
                    res["frontal"][nm] = fr(zz[sel], xx[sel])
                    uf |= res["frontal"][nm]
                    ul |= la(zz[sel], yy[sel])
            un = "12th rib" if k == 12 else f"{ap} {ordn(k)} rib"
            res["frontal"][un] = uf; res["lateral"][un] = ul
    return res
