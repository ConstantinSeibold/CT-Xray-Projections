import os
import glob
from PIL import Image
import SimpleITK as sitk
import numpy as np
from skimage.color import label2rgb
from skimage.morphology import label
from pqdm.processes import pqdm
import torch
import fid_score



# Function to read an image from a given path
def read_image(path):
    """
    Reads an image from the given path.

    Parameters:
        path (str): The path to the image file.

    Returns:
        images (SimpleITK.Image): The image object.
        images_array (np.ndarray): The image data as a numpy array.
    """
    images = sitk.ReadImage(path)
    images_array = sitk.GetArrayFromImage(images)
    images_array = np.clip(images_array, -1024, 3071)
    return images, images_array

# Function to normalize pixel values to [0, 255]
def fin(x):
    """
    Normalizes pixel values to the range [0, 255].

    Parameters:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Normalized array.
    """
    return ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)

# Function to display a large set of images
def display_large_set(n_imgs, path='/hkfs/work/workspace/scratch/pc8928-xray_training/openi/custom_annotation/OpenI/NLMCXR_512/'):
    """
    Display a large set of images as a single large image grid.

    Args:
    - n_imgs (int): Number of images to display in the grid.
    - path (str): Path to the directory containing the images.

    Returns:
    - numpy.ndarray: Concatenated image grid.
    """
    files = glob.glob(os.path.join(path, '*'))
    out = []
    for i in range(n_imgs):
        tmp = []
        for j in range(n_imgs):
            tmp += [np.array(Image.open(files[np.random.randint(len(files))]).convert("RGB"))]
        tmp = np.concatenate(tmp, 1)
        out += [tmp]
    out = np.concatenate(out, 0)
    return out 

# Color palette for segmentation masks
colors = [
    [np.random.uniform(), np.random.uniform(), np.random.uniform()]
    for i in range(200)
]

# Function to display an image with its segmentation mask
def display(image, mask=None, index=0, axis=0):
    """
    Display an image with its segmentation mask.

    Args:
    - image (numpy.ndarray): Input image.
    - mask (numpy.ndarray): Segmentation mask.
    - index (int): Index of the slice to display.
    - axis (int): Axis along which to display slices.

    Returns:
    - numpy.ndarray: Concatenated image with mask.
    """
    image = (image - image.min()) / (image.max() - image.min())
    
    if mask is None:
        mask = np.zeros(image.shape)
        
    if axis == 0:
        c = label2rgb(mask[::-1], image[::-1], colors=colors, bg_label=0)[index]
        r = np.stack([(image[::-1])[index]] * 3, -1)
    elif axis == 1:
        c = label2rgb(mask[::-1], image[::-1], colors=colors, bg_label=0)[:, index]
        r = np.stack([(image[::-1])[:, index]] * 3, -1)
    else:
        c = label2rgb(mask[::-1], image[::-1], colors=colors, bg_label=0)[:, :, index]
        r = np.stack([(image[::-1])[:, :, index]] * 3, -1)
    return (np.concatenate([c, r], 1) * 255).astype(np.uint8)

# Function to display a 2D image with its segmentation mask
def display2d(image, mask=None, index=0):
    """
    Display a 2D image with its segmentation mask.

    Args:
    - image (numpy.ndarray): Input image.
    - mask (numpy.ndarray): Segmentation mask.
    - index (int): Index of the slice to display.

    Returns:
    - numpy.ndarray: Concatenated image with mask.
    """
    image = (image - image.min()) / (image.max() - image.min())
    c = label2rgb(mask[::-1][index], image[::-1][index], colors=colors, bg_label=0)
    r = np.stack([(image[::-1])[index]] * 3, -1)
    return (np.concatenate([c, r], 1) * 255).astype(np.uint8)

# Function to calculate Fréchet Inception Distance (FID) for a given directory
def get_fid(dir_, target_dir='/hkfs/work/workspace/scratch/pc8928-xray_training/openi/custom_annotation/OpenI/NLMCXR_512/'):
    """
    Calculate Fréchet Inception Distance (FID) for images in a directory.

    Args:
    - dir_ (str): Path to the directory containing images.
    - target_dir (str): Path to the target directory.

    Returns:
    - tuple: FID score for frontal and lateral images.
    """
    fid_frontal = fid_score.calculate_fid_given_paths(
        [os.path.join(dir_, '1/'), target_dir],    
        256,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        2048,
        2,
    )
    fid_lateral = fid_score.calculate_fid_given_paths(
        [os.path.join(dir_, '2/'), target_dir],    
        256,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        2048,
        2,
    )
    return fid_frontal, fid_lateral

# Function to process a single file
def process_file(f, transform_function, cur_out_dir, axis):
    """
    Process a single file.

    Args:
    - f (str): Path to the input file.
    - transform_function (function): Transformation function.
    - cur_out_dir (str): Output directory.
    - axis (int): Axis along which to perform the transformation.
    """
    img, array = read_image(f)
    out_array = transform_function(array, axis)
    out_path = os.path.join(cur_out_dir, str(axis), f.split('/')[-1].replace('.nii.gz', '.png'))
    Image.fromarray(fin(out_array)).resize((512, 512), Image.LANCZOS).save(out_path)

# Function to process a folder of files
def process_folder(files, transform_function, cur_out_dir, axis, jobs=64):
    """
    Process a folder of files.

    Args:
    - files (list): List of input files.
    - transform_function (function): Transformation function.
    - cur_out_dir (str): Output directory.
    - axis (int): Axis along which to perform the transformation.
    - jobs (int): Number of parallel jobs to run.
    """

    if os.path.isdir(files):
        files = glob.glob(os.path.join(files, "**/*.nii.gz"), recursive=True)
    else:
        assert type(files) == list, "provide files either as directory or list of files"

    out_dir = os.path.join(cur_out_dir, str(axis))
    os.makedirs(out_dir, exist_ok=True)
    args = [{
            'f': f, 
            'transform_function': transform_function,
            'cur_out_dir': cur_out_dir,
            'axis': axis            
           } for f in files]
    pqdm(args, process_file, n_jobs=jobs, argument_type='kwargs')    

# Function to process a single file with resizing
def process_file_resize(f, transform_function, cur_out_dir, size_h, size_w, axis):
    """
    Process a single file with resizing.

    Args:
    - f (str): Path to the input file.
    - transform_function (function): Transformation function.
    - cur_out_dir (str): Output directory.
    - size_h (int): Height after resizing.
    - size_w (int): Width after resizing.
    - axis (int): Axis along which to perform the transformation.
    """
    img, array = read_image(f)
    out_array = transform_function(array, axis, size_h, size_w)
    out_path = os.path.join(cur_out_dir, str(axis), f.split('/')[-1].replace('.nii.gz', '.png'))
    Image.fromarray(fin(out_array)).resize((512, 512), Image.LANCZOS).save(out_path)
    print('saved {}'.format(out_path))

# Function to process a folder of files with resizing
def process_folder_resize(files, transform_function, cur_out_dir, size_h, size_w, axis, jobs=64):
    """
    Process a folder of files with resizing.

    Args:
    - files (list): List of input files.
    - transform_function (function): Transformation function.
    - cur_out_dir (str): Output directory.
    - size_h (int): Height after resizing.
    - size_w (int): Width after resizing.
    - axis (int): Axis along which to perform the transformation.
    - jobs (int): Number of parallel jobs to run.
    """
    out_dir = os.path.join(cur_out_dir, str(axis))
    os.makedirs(out_dir, exist_ok=True)
    args = [{
            'f': f, 
            'transform_function': transform_function,
            'cur_out_dir': cur_out_dir,
            'size_w': size_w,
            'size_h': size_h,
            'axis': axis,            
           } for f in files]
    pqdm(args, process_file_resize, n_jobs=jobs, argument_type='kwargs') 
