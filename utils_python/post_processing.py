import numpy as np
from skimage import morphology
from scipy.ndimage import zoom, label, binary_fill_holes
from .data_handling import read_niigz


def PostProcessNerve(img, ero_kernel, dil_kernel):
    # Opening
    img = morphology.binary_erosion(img, morphology.ball(ero_kernel))
    img = morphology.binary_dilation(img, morphology.ball(dil_kernel))


    # # Fill holes in 3D
    fill_hole1 = np.zeros_like(img, dtype=np.bool_)
    for frame_idx in range(img.shape[2]):  # top view
        fill_hole1[:, :, frame_idx] = binary_fill_holes(img[:, :, frame_idx])
    fill_hole2 = np.zeros_like(img, dtype=np.bool_)
    for frame_idx in range(img.shape[0]):  # front view
        fill_hole2[frame_idx, :, :] = binary_fill_holes(fill_hole1[frame_idx, :, :]) 
    fill_hole3 = np.zeros_like(img, dtype=np.bool_)
    for frame_idx in range(img.shape[1]):  # side view
        fill_hole3[:, frame_idx, :] = binary_fill_holes(fill_hole2[:, frame_idx, :])
    img = fill_hole3
    return img



def Remove_glandFP(vesselmask, glandmask):
    # Step 1: Identify fragments in vesselmask
    vessel_labeled, num_fragments = label(vesselmask)

    # Step 2: Create a copy of the vessel mask to modify
    filtered_vesselmask = np.copy(vesselmask)

    # Step 3: Loop through each vessel fragment
    for fragment_label in range(1, num_fragments + 1):
        # Create a mask for the current fragment
        fragment_mask = (vessel_labeled == fragment_label)
        
    # Total pixels in the current fragment
        fragment_size = np.sum(fragment_mask)
        
        # Count the overlapping pixels with glandmask where glandmask == 3
        overlap_count = np.sum(glandmask[fragment_mask] == 3)
        
        # Calculate the percentage of overlap
        overlap_percentage = overlap_count / fragment_size * 100
        
        # Remove the fragment if overlap > 10%
        if overlap_percentage > 20:
            filtered_vesselmask[fragment_mask] = 0

    # Output the updated vesselmask
    return filtered_vesselmask

def add_padding_z(img, padding=1):
    cut_len = padding*2
    padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)
    # Set the corners of the padding to 0
    padded_img[cut_len:-cut_len, :padding, :] = 1  # Top
    padded_img[cut_len:-cut_len, -padding:, :] = 1  # bottom
    padded_img[:padding, cut_len:-cut_len, :] = 1  # left
    padded_img[-padding:, cut_len:-cut_len, :] = 1  # right
    return padded_img

def remove_padding_z(padded_img, padding_width=1):
    return padded_img[padding_width:-padding_width, padding_width:-padding_width, :]

def holefilling_3D(img):
    fill_hole1 = np.zeros_like(img, dtype=np.bool_)
    for frame_idx in range(fill_hole1.shape[0]):  # front view
        fill_hole1[frame_idx, :, :] = binary_fill_holes(img[frame_idx, :, :]) 

    fill_hole2 = np.zeros_like(fill_hole1, dtype=np.bool_)
    for frame_idx in range(fill_hole2.shape[1]):  # side view
        fill_hole2[:, frame_idx, :] = binary_fill_holes(fill_hole1[:, frame_idx, :])

    fill_hole2 = add_padding_z(fill_hole2)
    fill_hole3 = np.zeros_like(fill_hole2, dtype=np.bool_)
    for frame_idx in range(fill_hole3.shape[2]):  # top view
        fill_hole3[:, :, frame_idx] = binary_fill_holes(fill_hole2[:, :, frame_idx])
    fill_hole3 = remove_padding_z(fill_hole3)

    img = fill_hole3
    return img


def process_vessel_mask(vessel_mask_path, gland_mask_path, 
                        z_start=0, z_end=664, 
                        downsample_factor=(0.25, 0.25, 0.25), 
                        dil_kernel=2, ero_kernel=2):

    def crop_along_shortest_axis(mask, z_start, z_end):
        """Crop the mask along the shortest axis between z_start and z_end."""
        shape = mask.shape
        shortest_axis = shape.index(min(shape))  # Find the index of the shortest axis
        if shortest_axis == 0:
            mask = mask[z_start:z_end, :, :]
        elif shortest_axis == 1:
            mask = mask[:, z_start:z_end, :]
        else:
            mask = mask[:, :, z_start:z_end]
        axes_order = [i for i in range(3) if i != shortest_axis] + [shortest_axis]
        mask = np.transpose(mask, axes_order)
        return mask
        
    if z_end > 640:
        z_end = 640
    # Load and crop vessel mask
    vessel_mask = read_niigz(vessel_mask_path)
    vessel_mask = crop_along_shortest_axis(vessel_mask, z_start, z_end)
    vessel_mask = zoom(vessel_mask, downsample_factor, order=0)

    # Load and crop gland mask
    gland_mask = read_niigz(gland_mask_path)
    gland_mask = crop_along_shortest_axis(gland_mask, z_start, z_end)
    gland_mask = zoom(gland_mask, downsample_factor, order=0)

    # Remove gland false positives
    vessel_mask = Remove_glandFP(vessel_mask, gland_mask)

    # Erosion, hole-filling, and dilation
    vessel_mask = morphology.binary_dilation(vessel_mask, morphology.ball(dil_kernel))
    vessel_mask = holefilling_3D(vessel_mask)
    vessel_mask = morphology.binary_erosion(vessel_mask, morphology.ball(ero_kernel))
    
    return vessel_mask