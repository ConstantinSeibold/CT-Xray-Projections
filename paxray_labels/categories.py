import json

import numpy as np

VERTEBRAE = [f"vertebrae {p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 5)) for i in range(1, n + 1)]
RIBS = [f"rib {k} {s}" for k in range(1, 13) for s in ("left", "right")]
LUNG_LOBES = ["lung upper lobe left", "lung lower lobe left",
              "lung upper lobe right", "lung middle lobe right", "lung lower lobe right"]
HEART = ["heart", "heart tissue", "heart atrium left", "heart atrium right", "heart myocardium",
         "heart ventricle left", "heart ventricle right"]

CATEGORIES = VERTEBRAE + RIBS + [
    "rib cartilage", "sternum", "manubrium", "clavicle left", "clavicle right", "scapula left", "scapula right",
] + LUNG_LOBES + ["trachea", "bronchi"] + HEART + [
    "aorta", "pulmonary artery", "inferior vena cava", "mediastinal tissue", "esophagus",
    "abdominal tissue", "liver", "stomach", "small bowel", "duodenum", "pancreas", "kidney left", "kidney right",
    "breast left", "breast right",
]

# internal label ids; 0 is background
CATEGORY = {n: i + 1 for i, n in enumerate(CATEGORIES)}
assert len(CATEGORY) == len(CATEGORIES) == 84


def read_mapping(path):
    """{category: [label ids]} from a JSON file mapping category names to one label id or a list of ids."""
    with open(path) as f:
        mapping = json.load(f)
    out, seen = {}, set()
    for name, ids in mapping.items():
        if name not in CATEGORY:
            raise ValueError(f"unknown category '{name}' in {path}")
        ids = [ids] if isinstance(ids, int) else list(ids)
        for i in ids:
            if not isinstance(i, int) or i < 0:
                raise ValueError(f"invalid label id {i!r} for '{name}' in {path}")
            if i in seen:
                raise ValueError(f"label id {i} mapped to more than one category in {path}")
            seen.add(i)
        out[name] = ids
    return out


def load_mapping(path):
    """lookup table from the label ids of the input volumes to internal ids; unmapped ids -> 0."""
    mapping = read_mapping(path)
    lut = np.zeros(max((i for ids in mapping.values() for i in ids), default=0) + 1, np.uint8)
    for name, ids in mapping.items():
        lut[ids] = CATEGORY[name]
    return lut


def remap(a, lut):
    a = np.rint(a).astype(np.int64)
    out = np.zeros(a.shape, np.uint8)
    valid = (a >= 0) & (a < len(lut))
    out[valid] = lut[a[valid]]
    return out


def inverse_lut(path):
    """lookup table from internal ids back to the label ids of the mapping (first id of each category);
    background and unmapped categories -> 0, or the smallest id not used by the mapping if 0 is used."""
    mapping = read_mapping(path)
    used = {i for ids in mapping.values() for i in ids}
    bg = 0 if 0 not in used else min(set(range(len(used) + 1)) - used)
    inv = np.full(len(CATEGORIES) + 1, bg, np.int64)
    for name, ids in mapping.items():
        if ids:
            inv[CATEGORY[name]] = ids[0]
    return inv
