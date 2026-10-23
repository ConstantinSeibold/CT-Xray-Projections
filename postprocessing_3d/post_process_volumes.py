import argparse
import logging
import multiprocessing
import os
import sys

import cc3d
import numpy as np
import SimpleITK as sitk
from skspatial.objects import Plane, Points

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paxray_labels.categories import CATEGORY, VERTEBRAE, inverse_lut, load_mapping, remap  # noqa: E402
from paxray_labels.common import list_volumes  # noqa: E402

# volume axes: x right->left, y anterior->posterior, z inferior->superior

LEFT_RIGHT_PAIRS = [("kidney left", "kidney right"), ("breast left", "breast right")] \
    + [(f"rib {k} left", f"rib {k} right") for k in range(12, 0, -1)] \
    + [("clavicle left", "clavicle right"), ("scapula left", "scapula right")]

# categories of which all connected components are kept; all others keep their largest component
KEEP_FRAGMENTS = {"abdominal tissue", "mediastinal tissue", "small bowel", "duodenum", "breast left", "breast right",
                  "rib cartilage", "aorta", "inferior vena cava", "pulmonary artery", "heart tissue", "bronchi",
                  "manubrium"}

# minimum fragment volumes (ml) in the left/right merging and the vertebra merging
MERGE_MIN_ML, VERTEBRA_MIN_ML = 0.006, 0.012

C = CATEGORY


def n_voxels(ml, spacing):
    return max(int(np.floor(ml * 1000.0 / np.prod(spacing) + 0.5)), 1)


def get_logger():
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s'))
        logger.addHandler(handler)
    return logger


def _unique_counts(a):
    """np.unique(a, return_counts=True) for non-negative integer label arrays."""
    flat = np.ravel(a, order="K")
    bc = np.bincount(flat.astype(np.int64, copy=False)) if flat.size else np.zeros(0, np.int64)
    uv = np.flatnonzero(bc)
    return uv.astype(flat.dtype), bc[uv]


def _counts(cc):
    uv, uc = _unique_counts(cc)
    return list(zip(uv.tolist(), uc.tolist()))


def get_relevant_ccs(cc, threshold):
    """components with more than threshold voxels, without background and the largest component."""
    return sorted([(i, n) for i, n in _counts(cc) if n > threshold], key=lambda x: x[1], reverse=True)[2:]


def merge_cc_of_adjacent(cc_cur, cc_other, threshold):
    """largest component of the union of the components of cc_cur and the non-largest components of cc_other;
    None if cc_other has no non-largest component."""
    relevant = sorted([(i, n) for i, n in _counts(cc_cur) if n > threshold], key=lambda x: x[1], reverse=True)[1:]
    relevant_other = get_relevant_ccs(cc_other, threshold)
    if not relevant_other:
        return None
    pool = np.zeros(cc_cur.shape, bool)
    for i, _ in relevant_other:
        pool |= cc_other == i
    for i, _ in relevant:
        pool |= cc_cur == i
    cc_pool = cc3d.connected_components(pool)
    return cc_pool == sorted(_counts(cc_pool), key=lambda x: x[1], reverse=True)[1][0]


def split_right_left(img, spacing):
    """Relabels left/right categories by their side of the plane fitted through the vertebrae and the sternum."""
    logger = get_logger()
    img_mod = img.copy()
    # plane and side test in mm
    mm = np.asarray(spacing, float)
    median_points = [np.median(np.argwhere(img == C[n]), axis=0) * mm for n in VERTEBRAE + ["sternum"] if (img == C[n]).any()]
    if len(median_points) < 3:
        logger.warning("Fewer than 3 vertebrae or sternum present. No right left splitting applied")
        return img_mod
    plane = Plane.best_fit(Points(median_points))
    normal_vector, point_plane = plane.vector, plane.point
    if not 0.9 < np.abs(normal_vector[0]) < 1.1:
        logger.warning("Plane not perpendicular to the x axis. No right left splitting applied")
        return img_mod
    left, right = (1, -1) if np.sign(normal_vector[0]) == -1 else (-1, 1)
    threshold = n_voxels(MERGE_MIN_ML, spacing)

    for name_left, name_right in LEFT_RIGHT_PAIRS:
        label_left, label_right = C[name_left], C[name_right]
        points_left = np.argwhere(img == label_left)
        points_right = np.argwhere(img == label_right)
        wrong = np.sign(np.dot(point_plane - points_left * mm, normal_vector)) == right
        if wrong.any():
            img_mod[tuple(points_left[wrong].T)] = label_right
        wrong = np.sign(np.dot(point_plane - points_right * mm, normal_vector)) == left
        if wrong.any():
            img_mod[tuple(points_right[wrong].T)] = label_left
        # components cut at the plane are merged into the largest component
        cc_left = cc3d.connected_components(img_mod == label_left)
        cc_right = cc3d.connected_components(img_mod == label_right)
        merged = merge_cc_of_adjacent(cc_left, cc_right, threshold)
        if merged is not None:
            img_mod[merged] = label_left
        merged = merge_cc_of_adjacent(cc_right, cc_left, threshold)
        if merged is not None:
            img_mod[merged] = label_right
    return img_mod


def spine_adjacent_pairs(img, spacing):
    """Removes small vertebra fragments and merges fragments assigned to the adjacent vertebra."""
    threshold = n_voxels(VERTEBRA_MIN_ML, spacing)
    img_mod = img.copy()
    for i in range(1, len(VERTEBRAE)):
        current = C[VERTEBRAE[i]]
        above = C[VERTEBRAE[i - 1]]
        below = C[VERTEBRAE[i + 1]] if i + 1 < len(VERTEBRAE) else None
        cc_cur = cc3d.connected_components(img_mod == current)
        for idx, n in _counts(cc_cur):
            if n <= threshold:
                img_mod[cc_cur == idx] = 0
        cc_above = cc3d.connected_components(img_mod == above, connectivity=6)
        if get_relevant_ccs(cc_above, threshold):
            merged = merge_cc_of_adjacent(cc_cur, cc_above, threshold)
            if merged is not None:
                img_mod[merged] = current
        elif below is not None:
            cc_below = cc3d.connected_components(img_mod == below, connectivity=6)
            if get_relevant_ccs(cc_below, threshold):
                merged = merge_cc_of_adjacent(cc_cur, cc_below, threshold)
                if merged is not None:
                    img_mod[merged] = current
    return img_mod


def count_ribs(img):
    """Relabels the 24 largest rib components by side and cranio-caudal order."""
    logger = get_logger()
    img_mod = img.copy()
    rib_ids = [C[f"rib {k} {s}"] for k in range(1, 13) for s in ("left", "right")]
    cc = cc3d.connected_components(np.isin(img, rib_ids), connectivity=6)
    uv, uc = _unique_counts(cc)
    components = uv[np.argsort(uc)[::-1][1:25]]
    if not len(components):
        return img_mod
    pts = {c: np.argwhere(cc == c) for c in components}
    by_median = [c for _, c in sorted([(np.median(pts[c], axis=0), c) for c in components], key=lambda x: x[0][0])]
    by_min = [c for _, c in sorted([(np.min(pts[c], axis=0), c) for c in components], key=lambda x: x[0][0])]
    if set(by_median[:12]) != set(by_min[:12]):
        logger.warning("Ambiguous right left assignment of the rib components. No rib processing applied")
        return img_mod
    sides = {"right": by_median[:12], "left": by_median[12:]}
    order = {s: [c for _, c in sorted([(np.mean(pts[c], axis=0), c) for c in cs], key=lambda x: x[0][2])]
             for s, cs in sides.items()}
    for i in range(min(len(order["left"]), len(order["right"]))):
        for s in ("left", "right"):
            img_mod[cc == order[s][i]] = C[f"rib {12 - i} {s}"]
    return img_mod


def _main_component(m):
    ccs = cc3d.connected_components(m)
    uv, uc = _unique_counts(ccs)
    return ccs == uv[np.argsort(uc)[::-1][1]]


def bodypart_limitations(img):
    """Removes small bowel and rib cartilage above the lower cervical spine; bronchi above the lungs become trachea."""
    img_mod = img.copy()
    c67 = [C["vertebrae C6"], C["vertebrae C7"]]
    boundary_head = None
    if all((img == c).any() for c in c67):
        boundary_head = np.mean(np.array([np.median(np.argwhere(img == c), axis=0) for c in c67]), axis=0)[2]
    upper = [C["lung upper lobe left"], C["lung upper lobe right"]]
    upper_bound_lung = None
    if all((img == c).any() for c in upper):
        upper_bound_lung = np.mean([np.max(np.argwhere(_main_component(img == c)), axis=0) for c in upper], axis=0)[2]

    if upper_bound_lung is not None:
        vox = np.argwhere(img_mod == C["bronchi"])
        vox = vox[vox[:, 2] > upper_bound_lung]
        img_mod[tuple(vox.T)] = C["trachea"]
    if boundary_head is not None:
        for n in ("small bowel", "rib cartilage"):
            vox = np.argwhere(img_mod == C[n])
            vox = vox[vox[:, 2] > boundary_head]
            img_mod[tuple(vox.T)] = 0
    return img_mod


def supress_non_largest_components(img):
    new_background = np.zeros(img.shape, bool)
    for name, label in C.items():
        if name in KEEP_FRAGMENTS:
            continue
        label_cc = cc3d.connected_components(img == label, connectivity=6)
        uv, uc = _unique_counts(label_cc)
        dominant = uv[np.argsort(uc)[::-1][:2]]
        if len(dominant) >= 2:
            new_background |= ~((label_cc == dominant[0]) | (label_cc == dominant[1]))
    img_mod = img.copy()
    img_mod[new_background] = 0
    return img_mod


def process_file(args):
    file_path, target_path, lut, inv = args
    file_name = os.path.basename(file_path)
    logger = get_logger()
    logger.info(f"Processing {file_name}")
    try:
        # processed in LPS orientation (array x right->left, y anterior->posterior, z inferior->superior), written back in the input orientation
        src = sitk.ReadImage(file_path)
        orientation = sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(src.GetDirection())
        lps = sitk.DICOMOrient(src, "LPS")
        img = remap(sitk.GetArrayFromImage(lps).transpose(2, 1, 0), lut)
        img = split_right_left(img, lps.GetSpacing())
        img = spine_adjacent_pairs(img, lps.GetSpacing())
        img = count_ribs(img)
        img = bodypart_limitations(img)
        img = supress_non_largest_components(img)
        out = inv[img]
        out = out.astype(np.uint8 if out.max() < 256 else np.uint16 if out.max() < 65536 else np.int32)
        out_img = sitk.GetImageFromArray(np.ascontiguousarray(out.transpose(2, 1, 0)))
        out_img.CopyInformation(lps)
        sitk.WriteImage(sitk.DICOMOrient(out_img, orientation), target_path, useCompression=True)
        logger.info(f"Finished {file_name}")
    except Exception:
        logger.exception(f"Error when processing {file_name}")


def main(args):
    os.makedirs(args.target_dir, exist_ok=True)
    files = [p for p in list_volumes(args.source_dir).values()
             if not os.path.isfile(os.path.join(args.target_dir, os.path.basename(p)))]
    lut, inv = load_mapping(args.mapping), inverse_lut(args.mapping)
    tasks = [(p, os.path.join(args.target_dir, os.path.basename(p)), lut, inv) for p in files]
    if args.nof_jobs > 1:
        with multiprocessing.Pool(args.nof_jobs) as pool:
            pool.map(process_file, tasks)
    else:
        for t in tasks:
            process_file(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Post-process 3D label volumes.')
    parser.add_argument("--source_dir", type=str, required=True, help="Directory with the 3D label volumes")
    parser.add_argument("--target_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--mapping", type=str, required=True, help="JSON file mapping category names to label ids")
    parser.add_argument("--nof_jobs", type=int, default=4, help="Number of parallel processes")
    main(parser.parse_args())
