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

## 3D rendering

`render_3d.py` renders a CT volume in 3D from the upper right, with standard HU bins (lung -950…-500, fat
-190…-30, muscle / soft tissue -29…150, bone ≥ 150) drawn as thin translucent surfaces. With a label map that uses
the [DAP Atlas](https://github.com/alexanderjaus/AtlasDataset) label ids it also writes one image per label group
(lung lobes and airways, heart and great vessels, rib cage, spine, abdominal organs, shoulder girdle and back
muscles), composited over a lightened copy of the CT rendering so every structure stays visible.

### bash

    # CT only
    python render_3d.py --ct /path/to/ct.nii.gz --out_dir /path/to/output

    # CT and DAP Atlas labels (label map on the CT grid)
    python render_3d.py --ct /path/to/ct.nii.gz --labels /path/to/atlas_prediction.nii.gz --out_dir /path/to/output

Optional: `--azimuth` (degrees from anterior towards patient left, default 35), `--elevation` (degrees above the
axial plane, default 25), `--device cuda|mps|cpu` (default: best available). Each image is written as PNG and PDF.
For a 512×512×159 CT with labels it takes about 25 s on an Apple M3 Max (MPS).

License

This project is licensed under the MIT License - see the LICENSE file for details.
