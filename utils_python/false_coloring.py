
import numpy as np
from skimage import exposure

# Hematoxylin and Eosin (HE) stain settings
HE_settings = {'nuclei': [0.17, 0.27, 0.105], 'cyto': [0.05, 1.0, 0.54]}

def getBackgroundLevels(image, threshold=50):
    """
    Estimate the background intensity level of an image.

    Args:
        image (numpy.ndarray): Input image array.
        threshold (int): Minimum intensity value to consider as foreground.

    Returns:
        tuple: (high value threshold, estimated background level)
    """
    image_DS = np.sort(image, axis=None)
    foreground_vals = image_DS[np.where(image_DS > threshold)]
    hi_val = foreground_vals[int(np.round(len(foreground_vals) * 0.95))]
    background = hi_val / 5
    return hi_val, background

def FC_rescale(image, ClipLow, ClipHigh):
    """
    Rescale image intensity with clipping and normalize to a fixed range.

    Args:
        image (numpy.ndarray): Input image.
        ClipLow (int): Lower clip value.
        ClipHigh (int): Upper clip value.

    Returns:
        numpy.ndarray: Rescaled image.
    """
    return exposure.rescale_intensity(np.clip(image, ClipLow, ClipHigh), out_range=(0, 10000))

def rapidFieldDivision(image, flat_field):
    """
    Perform field division for rapid false coloring.

    Args:
        image (numpy.ndarray): Input image.
        flat_field (numpy.ndarray): Precomputed flat field.

    Returns:
        numpy.ndarray: Corrected image.
    """
    return np.divide(image, flat_field, where=(flat_field != 0))

def rapidPreProcess(image, background, norm_factor):
    """
    Preprocess image by subtracting background and normalizing.

    Args:
        image (numpy.ndarray): Input image.
        background (float): Estimated background intensity.
        norm_factor (float): Normalization factor.

    Returns:
        numpy.ndarray: Preprocessed image.
    """
    tmp = image - background
    tmp[tmp < 0] = 0
    return (tmp ** 0.85) * (255 / norm_factor)

def rapidGetRGBframe(nuclei, cyto, nuc_settings, cyto_settings, k_nuclei, k_cyto):
    """
    Compute exponential false coloring for an RGB frame.

    Args:
        nuclei (numpy.ndarray): Nuclei image.
        cyto (numpy.ndarray): Cytoplasm image.
        nuc_settings (float): Nuclei intensity setting.
        cyto_settings (float): Cytoplasm intensity setting.
        k_nuclei (float): Multiplicative constant for nuclei.
        k_cyto (float): Multiplicative constant for cytoplasm.

    Returns:
        numpy.ndarray: Computed RGB frame.
    """
    tmp = nuclei * nuc_settings * k_nuclei + cyto * cyto_settings * k_cyto
    return (255 * np.exp(-1 * tmp)).astype(np.uint8)

def rapidFalseColor(nuclei, cyto, nuc_settings, cyto_settings,
                    nuc_normfactor=3000, cyto_normfactor=8000,
                    run_FlatField_nuc=False, run_FlatField_cyto=False,
                    nuc_bg_threshold=50, cyto_bg_threshold=50):
    """
    Perform rapid false coloring on nuclei and cytoplasm images.

    Args:
        nuclei (numpy.ndarray): Nuclei image.
        cyto (numpy.ndarray): Cytoplasm image.
        nuc_settings (list): Nuclei color settings.
        cyto_settings (list): Cytoplasm color settings.
        nuc_normfactor (int): Nuclei normalization factor.
        cyto_normfactor (int): Cytoplasm normalization factor.
        run_FlatField_nuc (bool): Apply flat field correction for nuclei.
        run_FlatField_cyto (bool): Apply flat field correction for cytoplasm.
        nuc_bg_threshold (int): Background threshold for nuclei.
        cyto_bg_threshold (int): Background threshold for cytoplasm.

    Returns:
        numpy.ndarray: RGB false-colored image.
    """
    nuclei = np.ascontiguousarray(nuclei, dtype=float)
    cyto = np.ascontiguousarray(cyto, dtype=float)

    k_nuclei = 1.0
    k_cyto = 1.0

    if not run_FlatField_nuc:
        k_nuclei = 0.08
        nuc_background = getBackgroundLevels(nuclei, threshold=nuc_bg_threshold)[1]
        nuclei = rapidPreProcess(nuclei, nuc_background, nuc_normfactor)

    if not run_FlatField_cyto:
        k_cyto = 0.012
        cyto_background = getBackgroundLevels(cyto, threshold=cyto_bg_threshold)[1]
        cyto = rapidPreProcess(cyto, cyto_background, cyto_normfactor)

    output_global = np.zeros((3, nuclei.shape[0], nuclei.shape[1]), dtype=np.uint8)
    for i in range(3):
        output_global[i] = rapidGetRGBframe(nuclei, cyto, nuc_settings[i], cyto_settings[i], k_nuclei, k_cyto)

    return np.moveaxis(output_global, 0, -1).astype(np.uint8)