import json
import os
import re
from types import SimpleNamespace

import SimpleITK as sitk

from paxray_labels.categories import CATEGORY, remap  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWS = ("frontal", "lateral")

_d = json.load(open(os.path.join(HERE, "paxray_classes.json")))
CLASSES = [_d[str(i)] for i in range(len(_d))]
IX = {n: i for i, n in enumerate(CLASSES)}

# channels that are empty in the lateral view: left/right classes except the heart chambers
NOT_LATERAL = {i for i, n in enumerate(CLASSES) if re.search(r"\b(left|right)\b", n) and not n.startswith("heart ")}


def stem(name):
    """file name without its extension (two extensions for compressed files, e.g. .nii.gz)."""
    base, ext = os.path.splitext(name)
    if ext.lower() in (".gz", ".bz2", ".zip"):
        base = os.path.splitext(base)[0]
    return base


def list_volumes(directory):
    """{stem: path} of the files in directory that SimpleITK can read."""
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        try:
            reader.ReadImageInformation()
        except RuntimeError:
            continue
        out[stem(name)] = path
    return out


def load_seg(path, lut):
    """label volume in internal ids as array (z superior->inferior, y anterior->posterior, x right->left), read-only."""
    a = remap(sitk.GetArrayFromImage(sitk.DICOMOrient(sitk.ReadImage(path), "LPS"))[::-1], lut)
    a.flags.writeable = False
    return a


def load_case(raw_path, pp_path, lut, pp_lut):
    s = sitk.DICOMOrient(sitk.ReadImage(raw_path), "LPS").GetSpacing()
    return SimpleNamespace(raw=load_seg(raw_path, lut), pp=load_seg(pp_path, pp_lut), spacing=(s[2], s[1], s[0]))
