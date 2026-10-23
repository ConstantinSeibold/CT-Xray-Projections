import numpy as np

from paxray_labels import common
from paxray_labels.postprocess_2d import postprocess_2d
from paxray_labels.rules import bones, lungs, mediastinum, ribs, vessels

_ZONES = [f"{s}{z}" for s in ("", "right ", "left ")
          for z in ("upper zone lung", "mid zone lung", "lung base", "apical zone lung")]
FAMILIES = {
    "ribs": [f"{ap} {ribs.ordn(k)} rib{s}" for k in range(1, 12) for ap in ("anterior", "posterior")
             for s in ("", " right", " left")] + ["posterior 12th rib right", "posterior 12th rib left", "12th rib"],
    "bones": bones.V + list(bones.AGG) + ["sternum", "rib_cartilage", "clavicle left", "clavicle right", "clavicles",
                                         "scapula left", "scapula right", "scapulas"],
    "lungs": lungs.R3 + lungs.L2 + ["right lung", "left lung", "lung"] + _ZONES
             + ["diaphragm", "right hemidiaphragm", "left hemidiaphragm", "breast", "breast right", "breast left"],
    "mediastinum": ["cardiomediastinum", "upper mediastinum", "lower mediastinum", "anterior mediastinum",
                    "middle mediastinum", "posterior mediastinum", "heart"] + mediastinum.CHAMBERS,
    "vessels": ["aorta", "ascending aorta", "descending aorta", "aortic arch", "trachea", "tracheal bifurcation",
                "pulmonary artery", "inferior vena cava"] + vessels.ABDOMEN,
}
RAW_ABDOMEN = {("frontal", "stomach"), ("frontal", "kidney right"), ("lateral", "liver")}
SOURCE = {n: f for f, ns in FAMILIES.items() for n in ns}
assert sorted(SOURCE) == sorted(common.CLASSES) and sum(map(len, FAMILIES.values())) == len(common.CLASSES)


def derive_case(raw_path, pp_path, lut, pp_lut):
    """{view: bool array (159, 512, 512)} from the raw and post-processed 3D label volumes of one case.
    lut, pp_lut: lookup tables from the label ids of the volumes to internal ids (categories.load_mapping)."""
    case = common.load_case(raw_path, pp_path, lut, pp_lut)
    res = {"ribs": ribs.derive(case), "bones": bones.derive(case), "lungs": lungs.derive(case),
           "mediastinum": mediastinum.derive(case), "vessels": vessels.derive(case, RAW_ABDOMEN)}
    arrs = {v: np.zeros((len(common.CLASSES), 512, 512), bool) for v in common.VIEWS}
    for v in common.VIEWS:
        for n in common.CLASSES:
            m = res[SOURCE[n]][v].get(n)
            c = common.IX[n]
            if m is not None and (v == "frontal" or c not in common.NOT_LATERAL):
                arrs[v][c] = m
    arrs["frontal"], arrs["lateral"] = postprocess_2d(arrs["frontal"], arrs["lateral"])
    return arrs
