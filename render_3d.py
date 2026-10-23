"""3D rendering of a CT volume with discrete HU bins, optionally with label groups (one image per group).

    python render_3d.py --ct CT.nii.gz --out_dir OUT [--labels LABELS.nii.gz] [--azimuth 35 --elevation 25]
                        [--device cuda|mps|cpu]

Context: an orthographic emission-absorption ray cast, seen from the viewer's upper right, on a 0.8 mm isotropic grid.
Every body voxel is put into a standard HU bin (lung -950..-500, fat -190..-30 and muscle / soft tissue -29..150 as in
the Alberta body-composition protocol, bone >= 150). Opacity sits on the boundaries between bins (thin translucent
surfaces, bone also gets a light volume fill), shading uses the gradient of the smoothed HU volume as surface normal.
The CT table is removed with a body mask. HU values only; no labels are used for the context.

Labels (optional): label maps with the DAP Atlas label ids (Task901 / Task902 predictions, label_name.csv of the
AtlasDataset repo). Each group in GROUPS is ray cast as shaded translucent surfaces and composited over a lightened
copy of the context, so a structure stays visible wherever it sits in the volume; labels in front still partly cover
labels behind them.

Implementation: the rotated volumes are never materialised; each depth slab of the camera grid is sampled with
grid_sample (rolling window of three slabs for the depth gradient), and the context and all label groups are
composited in one pass. Resampling, smoothing, the median filter and the erosion run on the device (CUDA, Apple MPS
or CPU).
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.patches import Patch
from scipy import ndimage as ndi

plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9})

STEP = 0.8  # mm, isotropic render grid
# (name, lo, hi, colour, surface opacity per sample, interior opacity per mm). HU outside every bin is transparent.
BINS = [
    ("lung", -950, -500, "#9cc3e4", 0.20, 0.000),
    ("fat", -190, -30, "#f3d9a4", 0.015, 0.000),
    ("muscle / soft tissue", -29, 150, "#e8907e", 0.040, 0.000),
    ("bone", 150, None, "#3c4f65", 0.22, 0.030),
]
SKIN = ("skin", "#e2c9a8", 0.035)  # body / air boundary
NB = len(BINS)
GAP = NB + 1  # inside the body but in no bin


def rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)], np.float32)


LUT_RGB = np.vstack([rgb(SKIN[1])] + [rgb(b[3]) for b in BINS] + [np.zeros(3, np.float32)])
LUT_S = np.array([SKIN[2]] + [b[4] for b in BINS] + [0.0], np.float32)
LUT_I = np.array([0.0] + [b[5] for b in BINS] + [0.0], np.float32)


def cmap_seq(name, n, lo=0.15, hi=0.9):
    return [np.array(colormaps[name](x)[:3]) for x in np.linspace(lo, hi, n)]


def hexs(cs):
    return [rgb(c) for c in cs]


# DAP Atlas label ids. Note: in the Task901 predictions we inspected, ids 67/68 ("costa 1" in label_name.csv) sit at
# the caudal end of the rib cage and 89/90 ("costa 12") at the cranial end; the names are kept as in label_name.csv.
RIBS = [(67 + 2 * i, 68 + 2 * i) for i in range(12)]
VERTEBRAE = [(43 + i, f"C{i + 1}") for i in range(7)] + [(50 + i, f"T{i + 1}") for i in range(12)] + \
            [(62 + i, f"L{i + 1}") for i in range(5)]
GROUPS = [
    ("lungs", "Lung lobes and airways",
     [(124, "upper lobe right"), (123, "middle lobe right"), (122, "lower lobe right"), (121, "upper lobe left"),
      (120, "lower lobe left"), (126, "trachea"), (125, "bronchi")],
     hexs(["#4e79a7", "#76b7b2", "#2f6f8f", "#a0cbe8", "#59a14f", "#e15759", "#f28e2b"])),
    ("heart", "Heart and great vessels",
     [(110, "ventricle left"), (111, "ventricle right"), (106, "atrium left"), (108, "atrium right"),
      (109, "myocardium"), (114, "aorta"), (127, "pulmonary artery"), (117, "inferior vena cava")],
     hexs(["#c0392b", "#2e86c1", "#e67e73", "#85c1e9", "#b07aa1", "#d4a017", "#5dade2", "#1f618d"])),
    ("ribcage", "Rib cage (costa 1–12), sternum, cartilage",
     [((l, r), f"costa {i + 1}") for i, (l, r) in enumerate(RIBS)] + [(92, "sternum"), (91, "rib cartilage")],
     cmap_seq("viridis", 12) + hexs(["#e15759", "#bab0ac"])),
    ("spine", "Vertebrae and spinal canal",
     VERTEBRAE + [(24, "spinal canal")],
     cmap_seq("plasma", len(VERTEBRAE), 0.05, 0.9) + hexs(["#59a14f"])),
    ("abdomen", "Abdominal organs",
     [(13, "liver"), (26, "spleen"), (7, "stomach"), (14, "pancreas"), (15, "kidney left"), (16, "kidney right"),
      (6, "esophagus"), (10, "colon"), (8, "small bowel")],
     hexs(["#9c6644", "#7b4b94", "#f2a65a", "#e9c46a", "#e76f51", "#f4a261", "#2a9d8f", "#577590", "#90be6d"])),
    ("shoulder_back", "Shoulder girdle and back muscles",
     [(95, "scapula left"), (96, "scapula right"), (93, "clavicle left"), (94, "clavicle right"),
      (40, "autochthon left"), (41, "autochthon right")],
     hexs(["#4e79a7", "#a0cbe8", "#f28e2b", "#ffbe7d", "#e15759", "#ff9d9a"])),
]
SURF_A, INNER_A = 0.62, 0.02  # label surface opacity per sample, interior per mm
FADE = 0.45  # context contrast kept under the labels

T0 = time.perf_counter()


def tick(msg):
    print(f"{time.perf_counter() - T0:6.1f}s  {msg}", flush=True)


def pick_device(want=None):
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gauss3d(v, sigma):
    """Separable Gaussian (truncate 4 sigma, edge padding, as scipy mode='nearest') on a (1, C, X, Y, Z) tensor.

    Written as a weighted sum of shifted copies instead of conv3d: on MPS, conv3d on 412 x 412 cross-sections
    returns wrong values (checked against CPU / scipy), elementwise ops do not.
    """
    r = int(4 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    for dim in (2, 3, 4):
        n = v.shape[dim]
        idx = torch.arange(-r, n + r, device=v.device).clamp(0, n - 1)
        p = v.index_select(dim, idx)
        out = torch.zeros_like(v)
        for i, w in enumerate(k):
            out += float(w) * p.narrow(dim, i, n)
        v = out
    return v


def median3(o, chunk=24):
    """3x3x3 median of a (X, Y, Z) small-integer tensor, chunked along X."""
    X = o.shape[0]
    p = F.pad(o[None, None].float(), (1, 1, 1, 1, 1, 1), mode="replicate")[0, 0]
    out = torch.empty_like(o)
    for x0 in range(0, X, chunk):
        x1 = min(X, x0 + chunk)
        blk = p[x0:x1 + 2]
        st = torch.stack([blk[a:a + x1 - x0, b:b + o.shape[1], c:c + o.shape[2]]
                          for a in range(3) for b in range(3) for c in range(3)])
        out[x0:x1] = st.median(0).values.to(o.dtype)
    return out


def body_mask(hu):
    m = ndi.binary_opening(hu > -400, iterations=2)
    lab, n = ndi.label(m)
    if n:
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    for z in range(m.shape[2]):
        m[:, :, z] = ndi.binary_fill_holes(m[:, :, z])
    return m


def camera(az, el):
    """Orthographic camera from the viewer's upper right: azimuth from anterior towards patient left, elevation
    above the axial plane. Returns (towards camera, screen up, screen right) in RAS voxel axes."""
    a, e = np.deg2rad(az), np.deg2rad(el)
    w = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    up = np.array([0, 0, 1.0]) - w[2] * w
    up /= np.linalg.norm(up)
    return w, up, np.cross(up, w)


def classify(sm, body, dev):
    """HU-bin label per iso voxel: 0 outside the body, 1..NB bins, GAP for body voxels outside every bin."""
    edges = torch.tensor([-950, -500, -190, -29.5, 150], device=dev)
    o = median3(torch.bucketize(sm, edges, right=True).to(torch.uint8))  # classes ordered by HU: median = despeckle
    lab = torch.tensor([GAP, 1, GAP, 2, 3, 4], device=dev, dtype=torch.uint8)[o.long()]
    del o
    lab[~body] = 0
    # partial volume at the skin would otherwise put a lung shell around the body
    inner = body.float()[None, None]
    for _ in range(5):
        inner = -F.max_pool3d(-inner, 3, 1, 1)
    lab[body & (inner[0, 0] < 0.5) & (lab == 1)] = GAP
    # isolated small "bone" components are noise or calcifications in soft tissue
    cc, n = ndi.label((lab == NB).cpu().numpy())
    if n:
        small = np.bincount(cc.ravel(), minlength=n + 1) < int(60 / STEP ** 3)
        small[0] = False
        lab[torch.from_numpy(small[cc]).to(dev)] = NB - 1
    return lab


def save_figure(img, out, title, legend, W, H):
    fig, ax = plt.subplots(figsize=(6.4, 6.6), facecolor="white")
    ax.imshow(img, interpolation="lanczos")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, color="#202124", loc="left")
    ax.legend(handles=[Patch(facecolor=cc, edgecolor="none", label=nm) for nm, cc in legend], frameon=False,
              fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=2 if len(legend) > 8 else 1,
              handlelength=1.1, handleheight=1.1, labelcolor="#3c4043", columnspacing=1.0)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.95)
    fig.savefig(out, dpi=250, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out[:-4] + ".pdf", facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="3D HU-bin rendering of a CT, optionally with DAP Atlas label groups.")
    ap.add_argument("--ct", required=True, help="CT volume (.nii.gz, HU)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--labels", help="label map on the CT grid with DAP Atlas label ids (.nii.gz)")
    ap.add_argument("--azimuth", type=float, default=35.0, help="degrees from anterior towards patient left")
    ap.add_argument("--elevation", type=float, default=25.0, help="degrees above the axial plane")
    ap.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    name = os.path.basename(args.ct).replace(".nii.gz", "").replace("_0000", "")
    dev = pick_device(args.device)
    tick(f"device {dev}")

    im = nib.as_closest_canonical(nib.load(args.ct))  # RAS: +x patient right, +y anterior, +z superior
    hu = im.get_fdata(dtype=np.float32)
    sp = np.array(im.header.get_zooms()[:3], float)
    body0 = body_mask(hu)
    atlas0 = None
    if args.labels:
        lim = nib.as_closest_canonical(nib.load(args.labels))
        atlas0 = np.asanyarray(lim.dataobj).astype(np.uint8)
        assert atlas0.shape == hu.shape, f"label map {atlas0.shape} is not on the CT grid {hu.shape}"
    tick("loaded, body mask")

    size = tuple(int(round(n * s / STEP)) for n, s in zip(hu.shape, sp))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    iso = F.interpolate(t(hu)[None, None], size=size, mode="trilinear", align_corners=True)
    body = F.interpolate(t(body0.astype(np.float32))[None, None], size=size, mode="nearest")[0, 0] > 0.5
    iso = torch.where(body[None, None], iso, torch.full_like(iso, -1000.0))
    sm = gauss3d(iso, 1.5)  # (1, 1, X, Y, Z); thin-slice noise would otherwise speckle the bins
    del iso
    lab = classify(sm[0, 0], body, dev)
    del body
    tick(f"iso grid {size}, classified")

    groups, legends, pals = [], [], []
    if atlas0 is not None:
        present = set(np.unique(atlas0).tolist())
        luts = []
        for key, title, items, colors in GROUPS:
            lut = np.zeros(256, np.uint8)
            leg = []
            for gi, (ids, nm) in enumerate(items, 1):
                ids = [i for i in (ids if isinstance(ids, tuple) else (ids,)) if i in present]
                if ids:
                    lut[ids] = gi
                    leg.append((nm, colors[gi - 1]))
            if leg:
                luts.append(lut)
                groups.append((key, title))
                legends.append(leg)
                pals.append(torch.tensor(np.vstack([np.zeros(3)] + [np.asarray(c, float) for c in colors]),
                                         device=dev, dtype=torch.float32))
    NG = len(groups)
    if NG:
        grp = F.interpolate(t(np.stack([lut[atlas0] for lut in luts]).astype(np.float32))[None], size=size,
                            mode="nearest")[0].to(torch.uint8)
        nrm = gauss3d((grp > 0).to(torch.float32)[None], 1.2)[0]  # smoothed masks for the label normals
        vols_f = torch.cat([sm[0], nrm], 0)[None]  # trilinear channels: HU, label normals
        vols_n = torch.cat([lab[None], grp], 0).float()[None]  # nearest channels: HU bin, label groups
        del grp, nrm
        tick(f"{NG} label groups")
    else:
        vols_f, vols_n = sm, lab[None, None].float()
    del sm, lab

    # output pixel (i, j) of slab k (k = 0 nearest to the camera) sits at iso index M (k, i, j) + off
    X, Y, Z = size
    c = np.array(size) // 2
    w, up, right = camera(args.azimuth, args.elevation)
    corners = np.array([[i, j, k] for i in (0, X) for j in (0, Y) for k in (0, Z)], float) - c
    ext = lambda v: (np.floor((corners @ v).min()), np.ceil((corners @ v).max()))
    d0, d1 = ext(w)
    u0, u1 = ext(up)
    r0, r1 = ext(right)
    D, H, W = int(d1 - d0) + 1, int(u1 - u0) + 1, int(r1 - r0) + 1
    M = np.stack([-w, -up, right], 1)
    off = c + d1 * w + u1 * up + r0 * right
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    base = (M[:, 1] * ii[..., None] + M[:, 2] * jj[..., None] + off).astype(np.float32)
    base_t = t(base)[..., [2, 1, 0]]  # grid_sample expects (z, y, x) = (W, H, D) order
    step_t = torch.tensor(M[:, 0][[2, 1, 0]].astype(np.float32), device=dev)
    scale = torch.tensor([2.0 / (Z - 1), 2.0 / (Y - 1), 2.0 / (X - 1)], device=dev)

    def sample(k):
        g = ((base_t + k * step_t) * scale - 1)[None, None]
        f = F.grid_sample(vols_f, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0]
        n_ = F.grid_sample(vols_n, g, mode="nearest", padding_mode="zeros", align_corners=True)[0, :, 0]
        f[0][(g.abs() > 1).any(-1)[0, 0]] = -1000.0
        return f, n_.round().to(torch.uint8)

    def neigh(L, Lprev, Lnext):
        Lp = F.pad(L[None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0]
        diff = (Lprev != L) | (Lnext != L)
        outer = (Lprev == 0) | (Lnext == 0)
        for n_ in (Lp[:-2, 1:-1], Lp[2:, 1:-1], Lp[1:-1, :-2], Lp[1:-1, 2:]):
            diff |= n_ != L
            outer |= n_ == 0
        return diff, outer

    def facing(v_prev, v, v_next):  # |cos| between the gradient and the view direction (headlight)
        gk = 0.5 * (v_next - v_prev)
        gi = torch.zeros_like(v)
        gj = torch.zeros_like(v)
        gi[1:-1] = 0.5 * (v[2:] - v[:-2])
        gj[:, 1:-1] = 0.5 * (v[:, 2:] - v[:, :-2])
        return gk.abs() / (torch.sqrt(gk ** 2 + gi ** 2 + gj ** 2) + 1e-6)

    lut_rgb, lut_s, lut_i = (torch.tensor(a, device=dev) for a in (LUT_RGB, LUT_S, LUT_I))
    acc_c = torch.zeros(H, W, 3, device=dev)
    acc_a = torch.zeros(H, W, device=dev)
    gc = torch.zeros(NG, H, W, 3, device=dev)
    ga = torch.zeros(NG, H, W, device=dev)
    zero = torch.zeros(H, W, device=dev)
    prev, cur = sample(0), sample(1)
    for k in range(1, D - 1):
        nxt = sample(k + 1)
        (fp, np_), (fc, nc), (fn, nn_) = prev, cur, nxt
        L = nc[0]
        if bool((L > 0).any()):
            diff, outer = neigh(L, np_[0], nn_[0])
            inside = L > 0
            key = torch.where(outer & inside, torch.zeros_like(L), L)  # body / air boundary samples are skin
            key = torch.where(L == NB, torch.full_like(L, NB), key).long()  # bone keeps its colour outside too
            a = torch.where(diff & inside & (L != GAP), lut_s[key], 1 - torch.exp(-lut_i[L.long()] * STEP))
            a = torch.where(inside, a, zero)
            nd = facing(fp[0], fc[0], fn[0])
            col = lut_rgb[key] * (0.55 + 0.45 * nd)[..., None] + (0.20 * nd ** 30)[..., None]
            tw = (1 - acc_a) * a
            acc_c += tw[..., None] * col
            acc_a += tw
        for g in range(NG):
            Lg = nc[1 + g]
            if not bool((Lg > 0).any()):
                continue
            diff, _ = neigh(Lg, np_[1 + g], nn_[1 + g])
            inside = Lg > 0
            a = torch.where(diff & inside, torch.full_like(zero, SURF_A),
                            torch.where(inside, torch.full_like(zero, 1 - np.exp(-INNER_A * STEP)), zero))
            nd = facing(fp[1 + g], fc[1 + g], fn[1 + g])
            col = pals[g][Lg.long()] * (0.35 + 0.65 * nd)[..., None] + (0.30 * nd ** 40)[..., None]
            tw = (1 - ga[g]) * a
            gc[g] += tw[..., None] * col
            ga[g] += tw
        prev, cur = cur, nxt
    context = (acc_c + (1 - acc_a)[..., None]).clamp(0, 1)
    faded = 1 - (1 - context) * FADE
    imgs = (gc + (1 - ga)[..., None] * faded[None]).clamp(0, 1).cpu().numpy()
    context = context.cpu().numpy()
    tick(f"rendered {D} slabs ({H}x{W})")

    bin_legend = [(f"{b[0]}   {b[1]} … {b[2]} HU" if b[2] is not None else f"{b[0]}   ≥ {b[1]} HU", rgb(b[3]))
                  for b in BINS]
    save_figure(context, os.path.join(args.out_dir, f"{name}_ct.png"), None, bin_legend, W, H)
    for g, (key, title) in enumerate(groups):
        save_figure(imgs[g], os.path.join(args.out_dir, f"{name}_{key}.png"), title, legends[g], W, H)
    tick(f"saved {1 + NG} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
