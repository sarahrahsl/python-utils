import os
import nibabel as nib
import cv2
import numpy as np
from scipy.ndimage import zoom


def nii_to_avi(input_path, framerate=40, downsample_factor=None, save_path=None):
    """Converts .nii.gz files to .avi format."""
    
    def process_file(nii_file, framerate, output_dir, downsample_factor):
        nii_data = nib.load(nii_file)
        volume = nii_data.get_fdata()

        if downsample_factor:
            volume = zoom(volume, downsample_factor, order=1)

        volume = ((volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255).astype(np.uint8)
        height, width, num_frames = volume.shape
        avi_file = os.path.join(output_dir, os.path.basename(nii_file).replace('.nii.gz', '.avi'))

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(avi_file, fourcc, framerate, (width, height), isColor=False)

        for i in range(num_frames):
            out.write(volume[:, :, i])

        out.release()
        print(f"Converted {nii_file} to {avi_file}")

    if os.path.isdir(input_path):
        nii_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith('.nii.gz')]
        output_dir = save_path or input_path
    elif input_path.endswith('.nii.gz'):
        nii_files = [input_path]
        output_dir = save_path or os.path.dirname(input_path)
    else:
        raise ValueError("Input path must be a .nii.gz file or a directory containing .nii.gz files.")

    os.makedirs(output_dir, exist_ok=True)

    for nii_file in nii_files:
        process_file(nii_file, framerate, output_dir, downsample_factor)
