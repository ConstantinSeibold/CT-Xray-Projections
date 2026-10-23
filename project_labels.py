import argparse
import multiprocessing
import os

import numpy as np

from paxray_labels.categories import load_mapping
from paxray_labels.common import list_volumes
from paxray_labels.derive import derive_case


def parse_arguments():
    parser = argparse.ArgumentParser(description='Derive PAXRay++ 2D labels from 3D label volumes.')
    parser.add_argument('--raw_dir', type=str, required=True, help='Directory with the raw 3D label volumes')
    parser.add_argument('--pp_dir', type=str, required=True, help='Directory with the post-processed 3D label volumes')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--mapping', type=str, required=True, help='JSON file mapping category names to label ids')
    parser.add_argument('--pp_mapping', type=str, help='Mapping of the post-processed volumes (default: --mapping)')
    parser.add_argument('--jobs', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--cases', type=str, nargs='+', help='Case names (default: all cases in raw_dir and pp_dir)')
    return parser.parse_args()


def process_case(args):
    case, raw_path, pp_path, out_dir, lut, pp_lut = args
    arrs = derive_case(raw_path, pp_path, lut, pp_lut)
    for view, m in arrs.items():
        np.save(os.path.join(out_dir, f"{case}_{view}.npy"), m)
    return case


def main():
    args = parse_arguments()
    raw, pp = list_volumes(args.raw_dir), list_volumes(args.pp_dir)
    cases = args.cases or sorted(set(raw) & set(pp))
    os.makedirs(args.out_dir, exist_ok=True)
    lut = load_mapping(args.mapping)
    pp_lut = load_mapping(args.pp_mapping) if args.pp_mapping else lut
    tasks = [(c, raw[c], pp[c], args.out_dir, lut, pp_lut) for c in cases]
    if args.jobs > 1:
        with multiprocessing.Pool(args.jobs) as pool:
            for c in pool.imap_unordered(process_case, tasks):
                print("done", c, flush=True)
    else:
        for t in tasks:
            print("done", process_case(t), flush=True)


if __name__ == '__main__':
    main()
