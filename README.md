# CT-Xray-Projections

This repository contains Python code for processing CT X-ray images. It includes utilities for image manipulation and various projection functions to enhance the images.
Installation

Clone this repository to your local machine:

### bash

    git clone https://github.com/yourusername/ct-xray-projection.git

## Requirements

    Python 3.x
    SimpleITK
    NumPy
    PIL (Pillow)
    scikit-image
    py-queue-pqdm
    torch
    scikit-learn

You can install the dependencies via pip:

### bash

    pip install -r requirements.txt

## Usage

-- Command-line Arguments

You can run the main script main.py with the following command-line arguments:

    --ct_dir: Path to the directory containing CT X-ray images.
    --out_dir: Path to the output directory.

For example:

### bash

    python main.py --ct_dir /path/to/ct_images --out_dir /path/to/output

## Functions

The utils.py file contains several utility functions for image processing, including:

    read_image: Reads an image from a given path.
    display_large_set: Displays a large set of images as a single large image grid.
    display: Displays an image with its segmentation mask.
    display2d: Displays a 2D image with its segmentation mask.
    get_fid: Calculates the Fréchet Inception Distance (FID) for images in a directory.
    process_folder: Processes a folder of files with a given transformation function.
    process_folder_resize: Processes a folder of files with resizing and a given transformation function.

The projections_essentials.py file contains essential functions for volume resampling and various projection techniques, including:

    resample_volume: Resamples a volume to a new spacing.
    Projection transfer functions:
        noexterior_transfer
        bone_noexterior_transfer
        bone_hist_transfer
        noexterior_hist_transfer
        hist_transfer
        matsu_transfer
        nonlinear_matsu_transfer
        bone_matsu_transfer
        sharpening_mastu_transfer
        hist_mastu_transfer
        numba_matsu_transfer
        bone_hist_resized_transfer
        noexterior_hist_resized_transfer
        hist_resized_transfer
        hist_before_resized_transfer

## Example
### python
    
    from utils import read_image, display
    from projections_essentials import hist_mastu_transfer
    
    # Read an image
    image_path = "/path/to/image.nii.gz"
    image, array = read_image(image_path)
    
    # Apply projection transfer function
    projection = hist_mastu_transfer(array, axis=0)
    
    # Display image with segmentation mask
    display(image, mask=projection)

## PAXRay++ labels

Derives the 159 PAXRay++ label classes (frontal and lateral view) from 3D label volumes of a CT segmentation model.

-- Input

    raw_dir/<case>.<ext>: 3D label volumes predicted by the segmentation model, in any format SimpleITK can read (e.g. .nii.gz, .nrrd, .mha).
    mapping.json: maps category names to the label ids of these volumes.

The category names are listed in CATEGORIES in paxray_labels/categories.py. A category maps to one label id or a list of ids; label ids that are not mapped are treated as background, categories that are not mapped as absent. Classes whose required categories are absent stay empty. Example:

### json

    {
        "vertebrae T4": 0,
        "rib 1 left": 1,
        "lung upper lobe right": 2,
        "heart": [3, 4],
        "aorta": 5
    }

-- Step 1: post-process the 3D labels (optional)

Any post-processing can be used instead, as long as its output is given with a mapping (--pp_mapping in step 2).

    --source_dir: Directory with the raw 3D label volumes.
    --target_dir: Output directory; the volumes are written in the input format with the label ids of the mapping.
    --mapping: Mapping file.
    --nof_jobs: Number of parallel processes (default 4).

### bash

    python postprocessing_3d/post_process_volumes.py --source_dir /path/to/raw --target_dir /path/to/pp --mapping mapping.json --nof_jobs 4

-- Step 2: derive the 2D labels

    --raw_dir: Directory with the raw 3D label volumes.
    --pp_dir: Directory with the post-processed 3D label volumes.
    --out_dir: Output directory.
    --mapping: Mapping file of the raw volumes.
    --pp_mapping: Mapping file of the post-processed volumes (default: --mapping).
    --jobs: Number of parallel processes (default 4).
    --cases: Optional list of case names, i.e. file names without extension (default: all cases present in raw_dir and pp_dir).

### bash

    python project_labels.py --raw_dir /path/to/raw --pp_dir /path/to/pp --out_dir /path/to/labels --mapping mapping.json --jobs 4

-- Output

    <out_dir>/<case>_frontal.npy, <out_dir>/<case>_lateral.npy: bool arrays of shape 159 x 512 x 512.
    The channel order is given by paxray_labels/paxray_classes.json.

## Quality control

Rule-based quality control of 3D label volumes. The rules are given in quality_control/qc_config.json and refer to the category names of paxray_labels/categories.py; rules only use categories that are mapped.

-- Input

    seg_dir/<case>.<ext>: 3D label volumes, in any format SimpleITK can read (e.g. .nii.gz, .nrrd, .mha).
    mapping.json: maps category names to the label ids of these volumes (as above).

-- Step 1: reference table (optional)

Size rules compare each category with the median volume of its largest connected component in accepted volumes. quality_control/qc_reference.csv is the table used for PAXRay++; for another segmentation model, build one from volumes accepted by visual inspection:

    --ids: File with one case name per line, i.e. file name without extension.

### bash

    python quality_control/qc_volumes.py reference /path/to/seg_dir reference.csv --config quality_control/qc_config.json --mapping mapping.json --ids accepted.txt --jobs 4

and set "reference" in the config to this file.

-- Step 2: evaluate the rules

    --config: QC config.
    --mapping: Mapping file of the volumes.
    --features: Optional; also writes the measured features, which can be re-evaluated with a changed config by "apply" instead of "run".
    --ids: Optional list of case names (default: all volumes in seg_dir).
    --jobs: Number of parallel processes (default 4).

### bash

    python quality_control/qc_volumes.py run /path/to/seg_dir qc.csv --config quality_control/qc_config.json --mapping mapping.json --jobs 4
    python quality_control/qc_volumes.py apply features.csv qc.csv --config quality_control/qc_config.json --mapping mapping.json

-- Output

    qc.csv: one row per volume with volume_id, qc_fail and qc_reason (the rules that fired and the offending categories, e.g. "missing structure: trachea; heart too small (< 0.2x reference): heart atrium left=0.118"). A volume that could not be read has an empty qc_fail and the error as qc_reason.

License

This project is licensed under the MIT License - see the LICENSE file for details.
