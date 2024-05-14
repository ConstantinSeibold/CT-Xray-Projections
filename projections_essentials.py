import SimpleITK as sitk
import numpy as np
import cc3d
from scipy.ndimage import binary_fill_holes
from skimage import exposure
from skimage.morphology import area_closing
from scipy import ndimage as ndi
from numba import njit



# Function to resample a volume to a new spacing
def resample_volume(volume, interpolator=sitk.sitkLinear, new_spacing=[1, 1, 1]):
    """
    Resamples a volume to a new spacing.

    Parameters:
        volume (SimpleITK.Image): The input volume.
        interpolator (int, optional): The interpolation method. Defaults to sitk.sitkLinear.
        new_spacing (list, optional): The new spacing. Defaults to [1, 1, 1].

    Returns:
        SimpleITK.Image: The resampled volume.
    """
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), new_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())



def noexterior_transfer(array, axis):
    """
    Transfer function for images without the exterior background.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp_array = array.copy()
    tmp_array = remove_backplate(tmp_array)
    return tmp_array[::-1].sum(axis)


def get_bones(array):
    """
    Get bones from the input image.

    Args:
        array (np.ndarray): Input image array.

    Returns:
        np.ndarray: Image array with bones.
    """
    bones = array.copy()
    bones[bones < 400] = 0
    bones[bones > 1600] = 0
    return bones


def remove_backplate(array):
    """
    Remove background from the input image.

    Args:
        array (np.ndarray): Input image array.

    Returns:
        np.ndarray: Image array with background removed.
    """
    tmp_array = array.copy()
    a = (tmp_array > -200)
    o = cc3d.connected_components(a)
    c, counts = np.unique(o, return_counts=True)

    out = (o == c[counts[1:].argmax() + 1])

    exterior = np.stack([binary_fill_holes(out[i]) for i in range(len(out))], 0)
    return tmp_array * exterior + (1 - exterior) * tmp_array.min()


def bone_noexterior_transfer(array, axis):
    """
    Transfer function for images with bones and no exterior background.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    body = remove_backplate(array.copy())
    bones = get_bones(body)
    out = body + bones * 2
    return out[::-1].sum(axis)


def bone_hist_transfer(array, axis):
    """
    Transfer function for histogram equalization with bones.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    body = remove_backplate(array.copy())
    bones = get_bones(body)
    out = body + bones
    out = out[::-1].sum(axis)
    return exposure.equalize_hist(out)


def noexterior_hist_transfer(array, axis):
    """
    Transfer function for histogram equalization without the exterior background.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp_array = array.copy()
    tmp_array = remove_backplate(tmp_array)
    tmp_array = tmp_array[::-1].sum(axis)
    return exposure.equalize_hist(tmp_array)


def hist_transfer(array, axis):
    """
    Transfer function for histogram equalization.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp_array = array.copy()
    tmp_array = tmp_array[::-1].sum(axis)
    return exposure.equalize_hist(tmp_array)


def matsu_transfer(array, axis):
    """
    Matsubara transfer function.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp = array[::-1].copy()
    mu_w = 0.00195

    integral = tmp.sum(axis)
    ave = integral / tmp.shape[axis]
    mu_ave = (ave * mu_w) / 1000 + mu_w
    mu_total = mu_ave * tmp.shape[axis]

    out = 1 - np.exp(-mu_total)
    return out


def nonlinear_matsu_transfer(array, axis):
    """
    Nonlinear Matsubara transfer function.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    def non_linear(inp):
        return 1 / (1 + np.exp(-a * (inp - b * inp.mean())))

    a = -4.0
    b = 2.0

    tmp = matsu_transfer(array, axis)
    out = non_linear(tmp)
    return 1 - out


def bone_matsu_transfer(array, axis):
    """
    Matsubara transfer function for images with bones.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    def non_linear(inp):
        return 1 / (1 + np.exp(-a * (inp - b * inp.mean())))

    a = -4.0
    b = 2.0

    tmp_all = matsu_transfer(array.copy(), axis)
    tmp_bones = matsu_transfer(get_bones(array.copy()), axis)

    out = non_linear(tmp_all) + non_linear(tmp_bones)
    return 1 - out


def sharpening_mastu_transfer(array, axis):
    """
    Sharpening Matsubara transfer function.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    w_u = 0.7
    tmp = bone_matsu_transfer(array, axis)
    blurred = ndi.uniform_filter(tmp, size=3)
    out = tmp + w_u * (tmp - blurred)
    return out


def hist_mastu_transfer(array, axis):
    """
    Histogram equalization with Matsubara transfer function.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp_array = sharpening_mastu_transfer(array, axis)
    return exposure.equalize_hist(tmp_array)


@njit
def numba_matsu_transfer(array, axis):
    """
    Numba-accelerated Matsubara transfer function.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Transformed array.
    """
    tmp = array[::-1].copy()
    mu_w = 0.00195

    integral = tmp.sum(axis)
    ave = integral / tmp.shape[axis]
    mu_ave = (ave * mu_w) / 1000 + mu_w
    mu_total = mu_ave * tmp.shape[axis]

    out = 1 - np.exp(-mu_total)
    return out

def bone_hist_resized_transfer(array, axis):
    """
    Histogram equalization with bones and resizing.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Resized and transformed array.
    """
    body = remove_backplate(array.copy())
    bones = get_bones(body)
    out = body + bones
    out = out[::-1].sum(axis)
    return exposure.equalize_hist(out)[:-64, 64:-64]


def noexterior_hist_resized_transfer(array, axis):
    """
    Histogram equalization without exterior background and resizing.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Resized and transformed array.
    """
    tmp_array = array.copy()
    tmp_array = remove_backplate(tmp_array)
    tmp_array = tmp_array[::-1].sum(axis)
    return exposure.equalize_hist(tmp_array)[:-64, 64:-64]


def hist_resized_transfer(array, axis, size_h=64, size_w=64):
    """
    Histogram equalization with resizing.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.
        size_h (int): Height of the resized image.
        size_w (int): Width of the resized image.

    Returns:
        np.ndarray: Resized and transformed array.
    """
    tmp_array = array.copy()
    tmp_array = tmp_array[::-1].sum(axis)
    return exposure.equalize_hist(tmp_array[:-size_h, size_w:-size_w])


def hist_before_resized_transfer(array, axis):
    """
    Histogram equalization before resizing.

    Args:
        array (np.ndarray): Input image array.
        axis (int): Axis along which to compute the sum.

    Returns:
        np.ndarray: Resized and transformed array.
    """
    tmp_array = array.copy()
    tmp_array = tmp_array[::-1].sum(axis)
    return exposure.equalize_hist(tmp_array)[:-64, 64:-64]