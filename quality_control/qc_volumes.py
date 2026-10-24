"""Rule-based quality control of 3D label volumes.

The rules are given in a config (qc_config.json) and refer to categories by name (paxray_labels/categories.py);
mapping.json maps these names to the label ids of the volumes. Rules only use mapped categories.

Per volume and category the script measures
  vox        number of voxels of the category (0 = missing)
  lcc_ml     volume of the largest connected component of the category (components below min_component_mm3 ignored;
             no such component = 0)
and evaluates the rules:
  component_count  number of connected components (>= min_component_mm3) of the union of a category group,
                   compared with a threshold                          e.g. too few ribs
  missing          any category of a group has no voxel
  size_ratio       lcc_ml / reference median below (op '<') or above (op '>') factor, for the categories present;
                   the rule fires when at least min_labels categories of the group do so
A volume fails QC when any rule fires; qc_reason lists the rules that fired with the offending categories.

Commands
  run        label volumes -> per-volume decisions (+ optional feature table)
  apply      feature table -> per-volume decisions (re-evaluate a changed config without reading the volumes)
  reference  label volumes of accepted cases -> reference table of the median lcc_ml per category
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cc3d
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paxray_labels.categories import CATEGORY, load_mapping, read_mapping, remap  # noqa: E402
from paxray_labels.common import list_volumes  # noqa: E402

CFG = None  # set in each worker


def load_config(path, mapping_path):
    cfg = json.load(open(path))
    groups = cfg.get("groups", {})
    mapped = set(read_mapping(mapping_path))

    def names(spec):  # "@group", a list of names / "@group" entries, or a single name
        if isinstance(spec, str):
            spec = [spec]
        out = []
        for s in spec:
            out += names(groups[s[1:]]) if s.startswith("@") else [s]
        return list(dict.fromkeys(out))

    rules = []
    for r in cfg["rules"]:
        all_names = names(r["labels"])
        unknown = [n for n in all_names if n not in CATEGORY]
        if unknown:
            raise SystemExit(f"rule {r['name']}: unknown categories {unknown}")
        r["_names"] = [n for n in all_names if n in mapped]
        if not r["_names"]:
            print(f"rule {r['name']}: no category mapped, rule skipped", file=sys.stderr)
            continue
        if len(r["_names"]) < len(all_names):
            print(f"rule {r['name']}: not mapped, ignored: {[n for n in all_names if n not in mapped]}", file=sys.stderr)
        rules.append(r)
    cfg["rules"] = rules
    cfg["_measured"] = list(dict.fromkeys(n for r in rules if r["type"] != "component_count" for n in r["_names"]))
    cfg["_lut"] = load_mapping(mapping_path)
    ref = cfg.get("reference")
    cfg["_ref_path"] = (ref if os.path.isabs(ref) else os.path.join(os.path.dirname(os.path.abspath(path)), ref)) \
        if ref else None
    cfg.setdefault("connectivity", 26)
    cfg.setdefault("min_component_mm3", 50)
    return cfg


def init(cfg):
    global CFG
    CFG = cfg


def measure(item):
    """Feature row of one volume: vox:<category>, lcc_ml:<category>, count:<rule> for component_count rules."""
    case, path = item
    f = {"volume_id": case}
    try:
        im = sitk.ReadImage(path)
        seg = remap(sitk.GetArrayFromImage(im), CFG["_lut"])
        vv = float(np.prod(im.GetSpacing()))  # mm3 per voxel
        conn, min_mm3 = CFG["connectivity"], CFG["min_component_mm3"]
        cnt = np.bincount(seg.ravel())
        cc = cc3d.connected_components(seg, connectivity=conn)  # components never join different categories
        size = np.bincount(cc.ravel())
        lab = np.zeros(len(size), dtype=np.int64)
        lab[cc.ravel()] = seg.ravel()
        keep = (size * vv >= min_mm3) & (np.arange(len(size)) > 0)
        lcc = pd.Series(size[keep] * vv / 1000).groupby(lab[keep]).max()  # ml
        for n in CFG["_measured"]:
            i = CATEGORY[n]
            f[f"vox:{n}"] = int(cnt[i]) if i < len(cnt) else 0
            f[f"lcc_ml:{n}"] = float(lcc.get(i, 0.0))
        for r in CFG["rules"]:
            if r["type"] == "component_count":
                m = np.isin(seg, [CATEGORY[n] for n in r["_names"]])
                c = cc3d.connected_components(m, connectivity=r.get("connectivity", conn))
                s = np.bincount(c.ravel())[1:]
                f[f"count:{r['name']}"] = int((s * vv >= r.get("min_component_mm3", min_mm3)).sum())
    except Exception as e:  # noqa: BLE001
        f["error"] = repr(e)[:300]
    return f


def compare(x, r, key):
    return x < r[key] if r["op"] == "<" else x > r[key]


def decide(feats, cfg, ref):
    """qc_fail and qc_reason per volume from a feature table."""
    reasons = [[] for _ in range(len(feats))]
    for r in cfg["rules"]:
        nm, t, d = r["name"], r["type"], r["description"]
        if t == "component_count":
            x = feats[f"count:{nm}"].to_numpy()
            for i in np.flatnonzero(compare(x, r, "threshold")):
                reasons[i].append(f"{d} ({int(x[i])} {r.get('unit', 'components')}, {r['op']} {r['threshold']})")
        elif t == "missing":
            v = feats[[f"vox:{n}" for n in r["_names"]]].fillna(0).to_numpy()
            for i, row in enumerate(v):
                if (row == 0).any():
                    reasons[i].append(f"{d}: " + ", ".join(n for n, z in zip(r["_names"], row) if z == 0))
        elif t == "size_ratio":
            med = ref.reindex(r["_names"])
            if med.isna().any():
                raise SystemExit(f"rule {nm}: no reference median for {list(med.index[med.isna()])}")
            q = feats[[f"lcc_ml:{n}" for n in r["_names"]]].fillna(0).to_numpy() / med.to_numpy()
            present = feats[[f"vox:{n}" for n in r["_names"]]].fillna(0).to_numpy() > 0
            hit = compare(q, r, "factor") & present  # missing categories are reported by the missing rule
            for i in np.flatnonzero(hit.sum(axis=1) >= r.get("min_labels", 1)):
                reasons[i].append(f"{d} ({r['op']} {r['factor']}x reference): "
                                  + ", ".join(f"{n}={v:.3f}" for n, v, h in zip(r["_names"], q[i], hit[i]) if h))
        else:
            raise SystemExit(f"unknown rule type {t}")
    out = pd.DataFrame({"volume_id": feats.volume_id.to_numpy(), "qc_fail": [bool(x) for x in reasons],
                        "qc_reason": ["; ".join(x) for x in reasons]}, dtype=object)
    if "error" in feats:
        err = feats["error"].notna().to_numpy()
        out.loc[err, "qc_fail"] = np.nan
        out.loc[err, "qc_reason"] = "error: " + feats.loc[err, "error"].astype(str).to_numpy()
    return out


def read_ref(cfg):
    if not cfg["_ref_path"]:
        return pd.Series(dtype=float)
    r = pd.read_csv(cfg["_ref_path"])
    return r.set_index(r.name.astype(str).str.strip()).median_lcc_ml


def volumes(seg_dir, ids):
    vols = list_volumes(seg_dir)
    if ids:
        wanted = [l.strip() for l in open(ids) if l.strip()]
        missing = [v for v in wanted if v not in vols]
        if missing:
            raise SystemExit(f"{len(missing)} volumes not found in {seg_dir}, e.g. {missing[:5]}")
        return [(v, vols[v]) for v in wanted]
    return list(vols.items())


def extract(cfg, seg_dir, ids, jobs):
    with Pool(jobs, initializer=init, initargs=(cfg,)) as p:
        rows = []
        for i, r in enumerate(p.imap(measure, volumes(seg_dir, ids), chunksize=2), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"{i} volumes", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "apply", "reference"])
    ap.add_argument("input", help="directory with the label volumes (run, reference) or feature table (apply)")
    ap.add_argument("out", help="output CSV")
    ap.add_argument("--config", required=True, help="QC config (JSON)")
    ap.add_argument("--mapping", required=True, help="JSON file mapping category names to label ids")
    ap.add_argument("--features", help="run: also write the feature table here")
    ap.add_argument("--ids", help="file with one volume name per line, i.e. file name without extension "
                                  "(default: all volumes in the directory)")
    ap.add_argument("--jobs", type=int, default=4, help="number of parallel processes")
    a = ap.parse_args()
    cfg = load_config(a.config, a.mapping)
    if a.command == "reference":
        if not a.ids:
            raise SystemExit("reference needs --ids with the accepted volumes")
        names = list(dict.fromkeys(n for r in cfg["rules"] if r["type"] == "size_ratio" for n in r["_names"]))
        cfg["_measured"] = names
        F = extract(cfg, a.input, a.ids, a.jobs)
        F = F[F.get("error", pd.Series(index=F.index, dtype=object)).isna()]
        rows = []
        for n in names:
            x = F[f"lcc_ml:{n}"]
            x = x[x > 0]
            # median on the log scale (geometric mean of the two middle values for an even count)
            rows.append({"name": n, "median_lcc_ml": float(np.exp(np.median(np.log(x)))) if len(x) else np.nan,
                         "n_volumes": int(len(x)), "n_reference": int(len(F))})
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"{a.out}: {len(rows)} categories from {len(F)} volumes")
        return
    feats = pd.read_csv(a.input) if a.command == "apply" else extract(cfg, a.input, a.ids, a.jobs)
    if a.features:
        feats.to_csv(a.features, index=False)
    out = decide(feats, cfg, read_ref(cfg))
    out.to_csv(a.out, index=False)
    q = out.qc_fail
    print(f"{a.out}: {len(out)} volumes, {int((q == True).sum())} fail, {int((q == False).sum())} pass, "  # noqa: E712
          f"{int(q.isna().sum())} errors")
    for r in cfg["rules"]:
        print(f"  {r['name']}: {int(out.qc_reason.str.contains(r['description'], regex=False).sum())}")


if __name__ == "__main__":
    main()
