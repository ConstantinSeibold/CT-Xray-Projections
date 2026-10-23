import numpy as np
from scipy import ndimage as ndi

from paxray_labels.common import IX

CL, CR, CLAV = IX["clavicle left"], IX["clavicle right"], IX["clavicles"]
APICAL = [(IX["apical zone lung"], IX["upper zone lung"]), (IX["right apical zone lung"], IX["right upper zone lung"]),
          (IX["left apical zone lung"], IX["left upper zone lung"])]


def _largest_with_neighbours(m, reach=10):
    lab, n = ndi.label(m)
    if n <= 1:
        return m.copy()
    a = ndi.sum(m, lab, range(1, n + 1))
    near = ndi.distance_transform_edt(lab != (np.argmax(a) + 1)) <= reach
    return np.isin(lab, np.unique(lab[near & m]))


def postprocess_2d(F, L):
    """F, L: frontal and lateral (159, 512, 512) masks."""
    F, L = F.copy(), L.copy()
    F[CL] = _largest_with_neighbours(F[CL])
    F[CR] = _largest_with_neighbours(F[CR])
    F[CLAV] = F[CL] | F[CR]
    rows = np.flatnonzero(F[CLAV].any(1))
    if len(rows):
        # apical zones above the lower edge of the clavicles, lateral clavicles within their row range
        top, cb = rows[0], rows[-1]
        keep = np.zeros(F.shape[1], bool)
        keep[: cb + 1] = True
        for a, _ in APICAL:
            F[a] &= keep[:, None]
        L[APICAL[0][0]] &= keep[:, None]
        lab, n = ndi.label(L[CLAV])
        if n:
            ok = [i + 1 for i, s in enumerate(ndi.find_objects(lab)) if s[0].start <= cb and s[0].stop - 1 >= top]
            L[CLAV] = np.isin(lab, ok)
    for a, u in APICAL:
        F[a] &= F[u]
    L[APICAL[0][0]] &= L[APICAL[0][1]]
    return F, L
