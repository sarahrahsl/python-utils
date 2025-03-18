import numpy as np
from scipy.ndimage import distance_transform_edt, zoom, label
from pathlib import Path
from .data_handling import read_niigz, read_tiff
from skimage.measure import regionprops, marching_cubes
from skimage.measure import label as sklabel


def crop_and_downsample3D(nerve_mask, cancer_mask, z_levels, zoom_factor = (0.25, 0.25, 0.25)):

    # Crop Z levels
    start, end = z_levels
    if end > 640:
        nerve_mask_cropped = nerve_mask[:, :, :round((640 - start) * 0.25)]
        cancer_mask_cropped = cancer_mask[:, :, start:640]
    else:
        nerve_mask_cropped = nerve_mask
        cancer_mask_cropped = cancer_mask[:, :, start:end]

    # Downsample cancer mask to match nerve mask
    cancer_mask_resized = zoom(cancer_mask_cropped, zoom_factor, order=0)

    # Ensure dimensions match
    print("nerve shape", nerve_mask_cropped.shape, "cancer shape", cancer_mask_resized.shape)
    if not nerve_mask_cropped.shape[2] == cancer_mask_resized.shape[2]:
        if nerve_mask_cropped.shape[2] > cancer_mask_resized.shape[2]:
            nerve_mask_cropped = nerve_mask_cropped[:, :, :cancer_mask_resized.shape[2]]
        if nerve_mask_cropped.shape[2] < cancer_mask_resized.shape[2]:
            cancer_mask_resized = cancer_mask_resized[:, :, :nerve_mask_cropped.shape[2]]

    if not nerve_mask_cropped.shape[1] == cancer_mask_resized.shape[1]:
        if nerve_mask_cropped.shape[1] > cancer_mask_resized.shape[1]:
            nerve_mask_cropped = nerve_mask_cropped[:, :cancer_mask_resized.shape[1], :]
        if nerve_mask_cropped.shape[1] < cancer_mask_resized.shape[1]:
            cancer_mask_resized = cancer_mask_resized[:, :nerve_mask_cropped.shape[1], :]

    if not nerve_mask_cropped.shape[0] == cancer_mask_resized.shape[0]:
        if nerve_mask_cropped.shape[0] > cancer_mask_resized.shape[0]:
            nerve_mask_cropped = nerve_mask_cropped[:cancer_mask_resized.shape[0], :, :]
        if nerve_mask_cropped.shape[0] < cancer_mask_resized.shape[0]:
            cancer_mask_resized = cancer_mask_resized[:nerve_mask_cropped.shape[0], :, :]
    
    # Ensure nerve mask is boolean and convert cancer mask from 0,255 to 0,1
    nerve_mask_cropped = nerve_mask_cropped.astype(bool)
    cancer_mask_resized = (cancer_mask_resized == 255).astype(bool)

    return nerve_mask_cropped, cancer_mask_resized


def compute_distance_maps(nerve_mask, pixelno_adj, pixelno_dis, tolerance=1):
    distance_map = distance_transform_edt(~nerve_mask)
    dilated_adj = distance_map <= pixelno_adj
    dilated_dis = distance_map <= pixelno_dis
    region_adj = dilated_adj & ~nerve_mask
    region_dis = dilated_dis & ~dilated_adj
    surface_adj = (distance_map >= pixelno_adj - tolerance) & (distance_map <= pixelno_adj + tolerance)
    surface_dis = (distance_map >= pixelno_dis - tolerance) & (distance_map <= pixelno_dis + tolerance)
    return region_adj, region_dis, surface_adj, surface_dis


def calculate_features(tiff_path, niigz_path, z_levels, pixelno_adj=20, pixelno_dis=40):

    # Load the nerve mask, cancer mask
    nerve_mask = read_niigz(niigz_path)  # Shape: (X, Y, Z)
    cancer_mask = read_tiff(tiff_path)  # Shape: (Z, X, Y)
    cancer_mask = np.transpose(cancer_mask, (1, 2, 0))  # Now (X, Y, Z)

    nerve_mask_cropped, cancer_mask_resized = crop_and_downsample3D(nerve_mask, cancer_mask, z_levels)

    # Label individual nerves
    labeled_nerves, num_labels = label(nerve_mask_cropped)
    print("number of nerve fragment:", num_labels)

    if num_labels == 0:
        stats = {key: np.nan for key in [
            "total_cancer_volume_adj", "total_cancer_volume_dis", "total_cancer_surface_adj", "total_cancer_surface_dis",
            "total_annulus_volume_adj", "total_annulus_volume_dis", "total_annulus_surface_adj", "total_annulus_surface_dis",
            "total_cancer_percentage_adj", "total_cancer_percentage_dis", "overall_cancer_gradient", "overall_percentage_gradient",
            "overall_cancer_invasion", "overall_percentage_invasion", "overall_cancer_surface_gradient", "overall_cancer_surface_invasion",
            "overall_percentage_surface_gradient", "overall_percentage_surface_invasion",
            
            "mean_cancer_adj", "med_cancer_adj", "min_cancer_adj", "max_cancer_adj", "sd_cancer_adj",
            "mean_cancer_dis", "med_cancer_dis", "min_cancer_dis", "max_cancer_dis", "sd_cancer_dis",
            "mean_percentage_adj", "med_percentage_adj", "min_percentage_adj", "max_percentage_adj", "sd_percentage_adj",
            "mean_percentage_dis", "med_percentage_dis", "min_percentage_dis", "max_percentage_dis", "sd_percentage_dis",
            "mean_gradient", "med_gradient", "min_gradient", "max_gradient", "sd_gradient",
            "mean_invasion", "med_invasion", "min_invasion", "max_invasion", "sd_invasion",
            "mean_percentage_gradient", "med_percentage_gradient", "min_percentage_gradient", "max_percentage_gradient", "sd_percentage_gradient",
            "mean_percentage_invasion", "med_percentage_invasion", "min_percentage_invasion", "max_percentage_invasion", "sd_percentage_invasion",
            "mean_surface_gradient", "med_surface_gradient", "min_surface_gradient", "max_surface_gradient", "sd_surface_gradient",
            "mean_surface_invasion", "med_surface_invasion", "min_surface_invasion", "max_surface_invasion", "sd_surface_invasion",
            "mean_percentage_surface_gradient", "med_percentage_surface_gradient", "min_percentage_surface_gradient", "max_percentage_surface_gradient", "sd_percentage_surface_gradient",
            "mean_percentage_surface_invasion", "med_percentage_surface_invasion", "min_percentage_surface_invasion", "max_percentage_surface_invasion", "sd_percentage_surface_invasion" 
        ]}

    elif num_labels > 0:

        # Compute distance map and annulus regions
        region_adj, region_dis, surface_adj, surface_dis = compute_distance_maps(nerve_mask_cropped, pixelno_adj, pixelno_dis)
        
        # Aggregating total cancer/annulus volumes
        total_nerve_volumes_adj = np.sum(region_adj)
        total_nerve_volumes_dis = np.sum(region_dis)
        total_cancer_volumes_adj = np.sum(cancer_mask_resized[region_adj]) if total_nerve_volumes_adj > 0 else np.nan
        total_cancer_volumes_dis = np.sum(cancer_mask_resized[region_dis]) if total_nerve_volumes_dis > 0 else np.nan
        total_cancer_percentage_adj = total_cancer_volumes_adj / total_nerve_volumes_adj
        total_cancer_percentage_dis = total_cancer_volumes_dis / total_nerve_volumes_dis
        total_nerve_surface_adj = np.sum(surface_adj) #surface
        total_nerve_surface_dis = np.sum(surface_dis) #surface
        total_cancer_surface_adj = np.sum(cancer_mask_resized[surface_adj]) if total_nerve_surface_adj > 0 else np.nan
        total_cancer_surface_dis = np.sum(cancer_mask_resized[surface_dis]) if total_nerve_surface_dis > 0 else np.nan
        total_cancer_surface_percentage_adj = total_cancer_surface_adj / total_nerve_surface_adj
        total_cancer_surface_percentage_dis = total_cancer_surface_dis / total_nerve_surface_dis

        # Compute ratios
        def safe_divide(a, b):
            return a / b if b > 0 else np.nan

        total_vol_gradient = safe_divide(total_cancer_volumes_dis, total_cancer_volumes_adj)
        total_vol_invasion = safe_divide(total_cancer_volumes_adj , total_cancer_volumes_dis)
        total_percentage_gradient = safe_divide(total_cancer_percentage_dis , total_cancer_percentage_adj)
        total_percentage_invasion = safe_divide(total_cancer_percentage_adj , total_cancer_percentage_dis)
        total_surface_gradient = safe_divide(total_cancer_surface_dis , total_cancer_surface_adj)
        total_surface_invasion = safe_divide(total_cancer_surface_adj, total_cancer_surface_dis)
        total_surface_percentage_gradient = safe_divide(total_cancer_surface_percentage_dis , total_cancer_surface_percentage_adj)
        total_surface_percentage_invasion = safe_divide(total_cancer_surface_percentage_adj , total_cancer_surface_percentage_dis)

        # Extract individual nerve annulus feature from here
        individual_cancer_volumes_adj = []
        individual_cancer_volumes_dis = []
        individual_cancer_surface_adj = []
        individual_cancer_surface_dis = []
        volumes_percentages_adj = []
        volumes_percentages_dis = []
        surfaces_percentages_adj = []
        surfaces_percentages_dis = []
        gradients = []
        percentage_gradients = []
        invasions = []
        percentage_invasions = []
        surface_gradients = []
        surface_invasions = []
        surface_percentage_gradients = []
        surface_percentage_invasions = []

        # Loop over each labeled nerve fragment
        for label_id in range(1, num_labels + 1):
            # Isolate nerve fragment and create corresponding annulus
            nerve_fragment = labeled_nerves == label_id
            if not np.sum(nerve_fragment) < 90000:
                region_fragment_adj, region_fragment_dis, surface_fragment_adj, surface_fragment_dis = \
                                        compute_distance_maps(nerve_fragment, pixelno_adj, pixelno_dis)

                # Calculate cancer volumes within regions for this nerve fragment
                cancer_volume_fragment_adj = np.sum(cancer_mask_resized[region_fragment_adj])
                cancer_volume_fragment_dis = np.sum(cancer_mask_resized[region_fragment_dis])
                cancer_surface_fragment_adj = np.sum(cancer_mask_resized[surface_fragment_adj])
                cancer_surface_fragment_dis = np.sum(cancer_mask_resized[surface_fragment_dis])
                percentage_volume_fragment_adj = cancer_volume_fragment_adj / np.sum(region_fragment_adj)
                percentage_volume_fragment_dis = cancer_volume_fragment_dis / np.sum(region_fragment_dis)
                percentage_surface_fragment_adj = cancer_surface_fragment_adj / np.sum(surface_fragment_adj)
                percentage_surface_fragment_dis = cancer_surface_fragment_dis / np.sum(surface_fragment_dis)
                gradient_fragment = cancer_volume_fragment_dis / cancer_volume_fragment_adj if cancer_volume_fragment_adj > 0 else -1
                percentage_gradient_fragment = percentage_volume_fragment_dis / percentage_volume_fragment_adj if percentage_volume_fragment_adj > 0 else -1
                invasion_fragment = percentage_volume_fragment_adj / percentage_volume_fragment_dis if percentage_volume_fragment_dis > 0 else -1
                percentage_invasion_fragment = percentage_volume_fragment_adj / percentage_volume_fragment_dis if percentage_volume_fragment_dis > 0 else -1
                surface_gradient_fragment = cancer_surface_fragment_dis / cancer_surface_fragment_adj if cancer_surface_fragment_adj > 0 else -1
                surface_invasion_fragment = cancer_surface_fragment_adj / cancer_surface_fragment_dis if cancer_surface_fragment_dis> 0 else -1
                percentage_surface_gradient_fragment = percentage_surface_fragment_dis / percentage_surface_fragment_adj if percentage_surface_fragment_adj > 0 else -1
                percentage_surface_invasion_fragment = percentage_surface_fragment_adj / percentage_surface_fragment_dis if percentage_surface_fragment_dis > 0 else -1
                # add percentage 
                individual_cancer_volumes_adj.append(cancer_volume_fragment_adj)
                individual_cancer_volumes_dis.append(cancer_volume_fragment_dis)
                individual_cancer_surface_adj.append(cancer_surface_fragment_adj)
                individual_cancer_surface_dis.append(cancer_surface_fragment_dis)
                volumes_percentages_adj.append(percentage_volume_fragment_adj)
                volumes_percentages_dis.append(percentage_volume_fragment_dis)
                surfaces_percentages_adj.append(percentage_surface_fragment_adj)
                surfaces_percentages_dis.append(percentage_surface_fragment_dis)
                if gradient_fragment != -1:
                    gradients.append(gradient_fragment)
                if percentage_gradient_fragment !=-1:
                    percentage_gradients.append(percentage_gradient_fragment)
                if invasion_fragment != -1:
                    invasions.append(invasion_fragment)
                if percentage_invasion_fragment !=-1:
                    percentage_invasions.append(percentage_invasion_fragment)
                if surface_gradient_fragment != -1:
                    surface_gradients.append(surface_gradient_fragment)
                if surface_invasion_fragment !=-1:
                    surface_invasions.append(surface_invasion_fragment)
                if percentage_surface_gradient_fragment != -1:
                    surface_percentage_gradients.append(percentage_surface_gradient_fragment)
                if percentage_surface_invasion_fragment !=-1:
                    surface_percentage_invasions.append(percentage_surface_invasion_fragment)

        # Aggregate statistics
        stats = {
            
            "total_cancer_volume_adj": total_cancer_volumes_adj,
            "total_cancer_volume_dis": total_cancer_volumes_dis,
            "total_cancer_surface_adj": total_cancer_surface_adj,
            "total_cancer_surface_dis": total_cancer_surface_dis,
            "total_annulus_volume_adj": total_nerve_volumes_adj,
            "total_annulus_volume_dis": total_nerve_volumes_dis,
            "total_annulus_surface_adj": total_nerve_surface_adj,
            "total_annulus_surface_dis": total_nerve_surface_dis,
            "total_cancer_percentage_adj": total_cancer_percentage_adj,
            "total_cancer_percentage_dis": total_cancer_percentage_dis,
            "overall_cancer_gradient": total_vol_gradient,
            "overall_percentage_gradient": total_percentage_gradient,
            "overall_cancer_invasion": total_vol_invasion,
            "overall_percentage_invasion": total_percentage_invasion,
            "overall_cancer_surface_gradient": total_surface_gradient,
            "overall_cancer_surface_invasion": total_surface_invasion,
            "overall_percentage_surface_gradient": total_surface_percentage_gradient,
            "overall_percentage_surface_invasion": total_surface_percentage_invasion,

            "mean_cancer_adj": np.mean(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan,
            "med_cancer_adj": np.median(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan,
            "min_cancer_adj": np.min(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan,
            "max_cancer_adj": np.max(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan,
            "sd_cancer_adj": np.std(individual_cancer_volumes_adj) if individual_cancer_volumes_adj else np.nan,

            "mean_cancer_dis": np.mean(individual_cancer_volumes_dis) if individual_cancer_volumes_dis else np.nan,
            "med_cancer_dis": np.median(individual_cancer_volumes_dis) if individual_cancer_volumes_dis else np.nan,
            "min_cancer_dis": np.min(individual_cancer_volumes_dis) if individual_cancer_volumes_dis else np.nan,
            "max_cancer_dis": np.max(individual_cancer_volumes_dis) if individual_cancer_volumes_dis else np.nan,
            "sd_cancer_dis": np.std(individual_cancer_volumes_dis) if individual_cancer_volumes_dis else np.nan,

            "mean_percentage_adj": np.mean(volumes_percentages_adj) if volumes_percentages_adj else np.nan,
            "med_percentage_adj": np.median(volumes_percentages_adj) if volumes_percentages_adj else np.nan,
            "min_percentage_adj": np.min(volumes_percentages_adj) if volumes_percentages_adj else np.nan,
            "max_percentage_adj": np.max(volumes_percentages_adj) if volumes_percentages_adj else np.nan,
            "sd_percentage_adj": np.std(volumes_percentages_adj) if volumes_percentages_adj else np.nan,

            "mean_percentage_dis": np.mean(volumes_percentages_dis) if volumes_percentages_dis else np.nan,
            "med_percentage_dis": np.median(volumes_percentages_dis) if volumes_percentages_dis else np.nan,
            "min_percentage_dis": np.min(volumes_percentages_dis) if volumes_percentages_dis else np.nan,
            "max_percentage_dis": np.max(volumes_percentages_dis) if volumes_percentages_dis else np.nan,
            "sd_percentage_dis": np.std(volumes_percentages_dis) if volumes_percentages_dis else np.nan,

            "mean_gradient": np.mean(gradients) if gradients else np.nan,
            "med_gradient": np.median(gradients) if gradients else np.nan,
            "min_gradient": np.min(gradients) if gradients else np.nan,
            "max_gradient": np.max(gradients) if gradients else np.nan,
            "sd_gradient": np.std(gradients) if gradients else np.nan,

            "mean_invasion": np.mean(invasions) if invasions else np.nan,
            "med_invasion": np.median(invasions) if invasions else np.nan,
            "min_invasion": np.min(invasions) if invasions else np.nan,
            "max_invasion": np.max(invasions) if invasions else np.nan,
            "sd_invasion": np.std(invasions) if invasions else np.nan,

            "mean_percentage_gradient": np.mean(percentage_gradients) if percentage_gradients else np.nan,
            "med_percentage_gradient": np.median(percentage_gradients) if percentage_gradients else np.nan,
            "min_percentage_gradient": np.min(percentage_gradients) if percentage_gradients else np.nan,
            "max_percentage_gradient": np.max(percentage_gradients) if percentage_gradients else np.nan,
            "sd_percentage_gradient": np.std(percentage_gradients) if percentage_gradients else np.nan,

            "mean_percentage_invasion": np.mean(percentage_invasions) if percentage_invasions else np.nan,
            "med_percentage_invasion": np.median(percentage_invasions) if percentage_invasions else np.nan,
            "min_percentage_invasion": np.min(percentage_invasions) if percentage_invasions else np.nan,
            "max_percentage_invasion": np.max(percentage_invasions) if percentage_invasions else np.nan,
            "sd_percentage_invasion": np.std(percentage_invasions) if percentage_invasions else np.nan,

            "mean_surface_gradient": np.mean(surface_gradients) if surface_gradients else np.nan,
            "med_surface_gradient": np.median(surface_gradients) if surface_gradients else np.nan,
            "min_surface_gradient": np.min(surface_gradients) if surface_gradients else np.nan,
            "max_surface_gradient": np.max(surface_gradients) if surface_gradients else np.nan,
            "sd_surface_gradient": np.std(surface_gradients) if surface_gradients else np.nan,

            "mean_surface_invasion": np.mean(surface_invasions) if surface_invasions else np.nan,
            "med_surface_invasion": np.median(surface_invasions) if surface_invasions else np.nan,
            "min_surface_invasion": np.min(surface_invasions) if surface_invasions else np.nan,
            "max_surface_invasion": np.max(surface_invasions) if surface_invasions else np.nan,
            "sd_surface_invasion": np.std(surface_invasions) if surface_invasions else np.nan,

            "mean_percentage_surface_gradient": np.mean(surface_percentage_gradients) if surface_percentage_gradients else np.nan, 
            "med_percentage_surface_gradient":  np.median(surface_percentage_gradients) if surface_percentage_gradients else np.nan,
            "min_percentage_surface_gradient":  np.min(surface_percentage_gradients) if surface_percentage_gradients else np.nan,
            "max_percentage_surface_gradient":  np.max(surface_percentage_gradients) if surface_percentage_gradients else np.nan,
            "sd_percentage_surface_gradient":  np.std(surface_percentage_gradients) if surface_percentage_gradients else np.nan,

            "mean_percentage_surface_invasion": np.mean(surface_percentage_invasions) if surface_percentage_invasions else np.nan,
            "med_percentage_surface_invasion":  np.median(surface_percentage_invasions) if surface_percentage_invasions else np.nan,
            "min_percentage_surface_invasion":  np.min(surface_percentage_invasions) if surface_percentage_invasions else np.nan,
            "max_percentage_surface_invasion":  np.max(surface_percentage_invasions) if surface_percentage_invasions else np.nan,
            "sd_percentage_surface_invasion":  np.std(surface_percentage_invasions) if surface_percentage_invasions else np.nan,
        }

    # Extract sample name
    p = Path(niigz_path)
    index = p.parts.index("UPenn_Clinical")
    sample_name = p.parts[index + 1]
    print(sample_name)

    # Compile result for this sample
    result = {
        "sample_name": sample_name,
        **stats
    }

    return result



def calculate_properties(tiff_path, niigz_path, z_levels, sliceno=None):
    # Load the nerve mask, cancer mask
    start, end = z_levels
    if sliceno is not None:
        nerve_mask_cropped = read_niigz(niigz_path, int(start/4+sliceno))  # Shape: (X, Y, Z)
        cancer_mask = read_tiff(tiff_path)  # Shape: (Z, X, Y)
        cancer_mask = np.transpose(cancer_mask, (1, 2, 0))  # Now (X, Y, Z)
        cancer_mask = cancer_mask[:, :, start+sliceno]
        cancer_mask_resized = zoom(cancer_mask, (0.25, 0.25), order=0)
    else:
        nerve_mask = read_niigz(niigz_path)  # Shape: (X, Y, Z)
        cancer_mask = read_tiff(tiff_path)  # Shape: (Z, X, Y)
        cancer_mask = np.transpose(cancer_mask, (1, 2, 0))  # Now (X, Y, Z)
        nerve_mask_cropped, cancer_mask_resized = crop_and_downsample3D(nerve_mask, cancer_mask, z_levels)

    tumor_nerve = nerve_mask_cropped & cancer_mask_resized
    stroma_nerve = nerve_mask_cropped & ~cancer_mask_resized
    labeled_mask, num_fragments = label(nerve_mask_cropped)
    labeled_tumor, num_tumor_fragments = label(tumor_nerve)
    labeled_stroma, num_stroma_fragments = label(stroma_nerve)
    
    # Compute properties for each segmentation
    stats_nerve = getObjectProperties(labeled_mask, num_fragments, "nerve")
    stats_tumor = getObjectProperties(labeled_tumor, num_tumor_fragments, "tumor")
    stats_stroma = getObjectProperties(labeled_stroma, num_stroma_fragments, "stroma")

    # Extract sample name
    p = Path(niigz_path)
    index = p.parts.index("UPenn_Clinical")
    sample_name = p.parts[index + 1]
    print(sample_name)

    # Compile result for this sample
    result = {
        "sample_name": sample_name,
        **stats_nerve, 
        **stats_tumor, 
        **stats_stroma
    }

    return result


def calculate_gland_properties(tiff_path, niigz_path, z_levels, sliceno=None):
    # Load the nerve mask, cancer mask
    start, end = z_levels
    if sliceno is not None:
        gland_mask_cropped = read_niigz(niigz_path, int(start+sliceno*4))  # Shape: (X, Y, Z)
        gland_mask_cropped = zoom(gland_mask_cropped, (0.25, 0.25), order=0)
        cancer_mask = read_tiff(tiff_path)  # Shape: (Z, X, Y)
        cancer_mask = np.transpose(cancer_mask, (1, 2, 0))  # Now (X, Y, Z)
        cancer_mask = cancer_mask[:, :, start+sliceno]
        cancer_mask_resized = zoom(cancer_mask, (0.25, 0.25), order=0)
    else:
        gland_mask = read_niigz(niigz_path)  # Shape: (X, Y, Z)
        gland_mask = gland_mask[:,:,start:end]
        gland_mask = zoom(gland_mask, (0.25, 0.25, 0.25), order=0)
        cancer_mask = read_tiff(tiff_path)  # Shape: (Z, X, Y)
        cancer_mask = np.transpose(cancer_mask, (1, 2, 0))  # Now (X, Y, Z)
        gland_mask_cropped, cancer_mask_resized = crop_and_downsample3D(gland_mask, cancer_mask, z_levels)

    # Separate lumen, stroma, and epithelium
    lu_mask = ((gland_mask_cropped == 1)).astype(int)
    st_mask = ((gland_mask_cropped == 2)).astype(int)
    ep_mask = ((gland_mask_cropped == 3)).astype(int)
    cancer_lu_mask = lu_mask & cancer_mask_resized
    nonca_lu_mask = lu_mask & ~cancer_mask_resized
    cancer_st_mask = st_mask & cancer_mask_resized
    nonca_st_mask = st_mask & ~cancer_mask_resized
    cancer_ep_mask = ep_mask & cancer_mask_resized
    nonca_ep_mask = ep_mask & ~cancer_mask_resized


    # Calculate total vol, surface, and convex hull for each mask + individual masks
    lumen_stats, total_lumen_vol, total_lumen_sa, total_lumen_chv = getTotalproperties(lu_mask, "lumen")
    stroma_stats, total_stroma_vol, total_stroma_sa, total_stroma_chv = getTotalproperties(st_mask, "stroma")
    epithelium_stats, total_epithelium_vol, total_epithelium_sa, total_epithelium_chv = getTotalproperties(ep_mask, "epithelium")

    cancer_lumen_stats, cancer_lumen_vol, cancer_lumen_sa, cancer_lumen_chv = getTotalproperties(cancer_lu_mask, "cancer_lumen")
    cancer_stroma_stats, cancer_stroma_vol, cancer_stroma_sa, cancer_stroma_chv = getTotalproperties(cancer_st_mask, "cancer_stroma")
    cancer_epithelium_stats, cancer_epithelium_vol, cancer_epithelium_sa, cancer_epithelium_chv = getTotalproperties(cancer_ep_mask, "cancer_epithelium")

    noncancer_lumen_stats, noncancer_lumen_vol, noncancer_lumen_sa, noncancer_lumen_chv = getTotalproperties(nonca_lu_mask, "noncancer_lumen")
    noncancer_stroma_stats, noncancer_stroma_vol, noncancer_stroma_sa, noncancer_stroma_chv = getTotalproperties(nonca_st_mask, "noncancer_stroma")
    noncancer_epithelium_stats, noncancer_epithelium_vol, noncancer_epithelium_sa, noncancer_epithelium_chv = getTotalproperties(nonca_ep_mask, "noncancer_epithelium")

    # Extract sample name
    p = Path(niigz_path)
    index = p.parts.index("UPenn_Clinical")
    sample_name = p.parts[index + 1]
    if sample_name == "For Jennifer":
        sample_name = p.parts[index + 2]
    print(sample_name)

    total_stats = {
        "sample_name": sample_name,
        "Total_lumen_volume": total_lumen_vol,
        "Total_stroma_volume": total_stroma_vol,
        "Total_epithelium_volume": total_epithelium_vol,
        "Total_lumen_surface_area": total_lumen_sa,
        "Total_stroma_surface_area": total_stroma_sa,
        "Total_epithelium_surface_area": total_epithelium_sa,
        "Total_lumen_epithelium_ratio": calculate_ratio(total_lumen_vol, total_epithelium_vol),
        "Total_stroma_epithelium_ratio": calculate_ratio(total_stroma_vol, total_epithelium_vol),
        "Total_lumen_SA_to_V_ratio": calculate_ratio(total_lumen_sa, total_lumen_vol),
        "Total_stroma_SA_to_V_ratio": calculate_ratio(total_stroma_sa, total_stroma_vol),
        "Total_epithelium_SA_to_V_ratio": calculate_ratio(total_epithelium_sa, total_epithelium_vol),
        "Total_lumen_solidity": calculate_ratio(total_lumen_vol, total_lumen_chv),
        "Total_stroma_solidity": calculate_ratio(total_stroma_vol, total_stroma_chv),
        "Total_epithelium_solidity": calculate_ratio(total_epithelium_vol, total_epithelium_chv),
        **lumen_stats,
        **stroma_stats,
        **epithelium_stats
    }

    cancer_stats = {
        "sample_name": sample_name,
        "cancer_lumen_volume": cancer_lumen_vol,
        "cancer_stroma_volume": cancer_stroma_vol,
        "cancer_epithelium_volume": cancer_epithelium_vol,
        "cancer_lumen_surface_area": cancer_lumen_sa,
        "cancer_stroma_surface_area": cancer_stroma_sa,
        "cancer_epithelium_surface_area": cancer_epithelium_sa,
        "cancer_lumen_epithelium_ratio": calculate_ratio(cancer_lumen_vol, cancer_epithelium_vol),
        "cancer_stroma_epithelium_ratio": calculate_ratio(cancer_stroma_vol, cancer_epithelium_vol),
        "cancer_lumen_SA_to_V_ratio": calculate_ratio(cancer_lumen_sa, cancer_lumen_vol),
        "cancer_stroma_SA_to_V_ratio": calculate_ratio(cancer_stroma_sa, cancer_stroma_vol),
        "cancer_epithelium_SA_to_V_ratio": calculate_ratio(cancer_epithelium_sa, cancer_epithelium_vol),
        "cancer_lumen_solidity": calculate_ratio(cancer_lumen_vol, cancer_lumen_chv),
        "cancer_stroma_solidity": calculate_ratio(cancer_stroma_vol, cancer_stroma_chv),
        "cancer_epithelium_solidity": calculate_ratio(cancer_epithelium_vol, cancer_epithelium_chv),
        **cancer_lumen_stats,
        **cancer_stroma_stats,
        **cancer_epithelium_stats
    }

    noncancer_stats = {
        "sample_name": sample_name,
        "noncancer_lumen_volume": noncancer_lumen_vol,
        "noncancer_stroma_volume": noncancer_stroma_vol,
        "noncancer_epithelium_volume": noncancer_epithelium_vol,
        "noncancer_lumen_surface_area": noncancer_lumen_sa,
        "noncancer_stroma_surface_area": noncancer_stroma_sa,
        "noncancer_epithelium_surface_area": noncancer_epithelium_sa,
        "noncancer_lumen_epithelium_ratio": calculate_ratio(noncancer_lumen_vol, noncancer_epithelium_vol),
        "noncancer_stroma_epithelium_ratio": calculate_ratio(noncancer_stroma_vol, noncancer_epithelium_vol),
        "noncancer_lumen_SA_to_V_ratio": calculate_ratio(noncancer_lumen_sa, noncancer_lumen_vol),
        "noncancer_stroma_SA_to_V_ratio": calculate_ratio(noncancer_stroma_sa, noncancer_stroma_vol),
        "noncancer_epithelium_SA_to_V_ratio": calculate_ratio(noncancer_epithelium_sa, noncancer_epithelium_vol),
        "noncancer_lumen_solidity": calculate_ratio(noncancer_lumen_vol, noncancer_lumen_chv),
        "noncancer_stroma_solidity": calculate_ratio(noncancer_stroma_vol, noncancer_stroma_chv),
        "noncancer_epithelium_solidity": calculate_ratio(noncancer_epithelium_vol, noncancer_epithelium_chv),
        **noncancer_lumen_stats,
        **noncancer_stroma_stats,
        **noncancer_epithelium_stats
    }

    return total_stats, cancer_stats, noncancer_stats


def calculate_surface_area(mask):
    """
    Compute the surface area of a 3D binary mask using the marching cubes algorithm.
    """
    if not np.any(mask):  
        return 0  # If mask is completely empty, surface area is 0
    
    mask = (mask > 0).astype(np.uint8)  # Ensure binary mask

    try:
        verts, faces, _, _ = marching_cubes(mask, level=0.5)  # Extract surface at 0.5 threshold
        surface_area = np.sum(np.linalg.norm(np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                                                      verts[faces[:, 2]] - verts[faces[:, 0]]), axis=1)) / 2
        return surface_area
    except ValueError:
        return 0  # If marching_cubes fails (e.g., fully solid mask), return 0

def calculate_ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0  # Avoid division by zero


def getTotalproperties(mask, prefix):

    stats = {}
    regions = regionprops(mask)

    volumes, convex_volumes, solidities= [], [], []

    for region in regions:
        volumes.append(region.area)
        convex_volumes.append(region.convex_area)
        solidities.append(region.solidity)

    for name, values in zip(
        ["volume", "convex_hull_volume", "solidity"], [volumes, convex_volumes, solidities]
    ):
       values = np.array(values, dtype=np.float64)  # Ensure numerical stability
        
    if values.size == 0:  # If no regions exist, fill with NaN or 0
        stats.update({
            f"{prefix}_{name}_mean": np.nan,
            f"{prefix}_{name}_median": np.nan,
            f"{prefix}_{name}_std": np.nan,
            f"{prefix}_{name}_min": np.nan,
            f"{prefix}_{name}_max": np.nan,
            f"{prefix}_{name}_sum": 0,  # Sum should be 0 if no regions exist
        })
    else:
        stats.update({
            f"{prefix}_{name}_mean": np.nanmean(values),
            f"{prefix}_{name}_median": np.nanmedian(values),
            f"{prefix}_{name}_std": np.nanstd(values),
            f"{prefix}_{name}_min": np.nanmin(values),
            f"{prefix}_{name}_max": np.nanmax(values),
            f"{prefix}_{name}_sum": np.nansum(values),
        })

    if mask.ndim == 3:  # 3D case
        total_SA_or_peri = calculate_surface_area(mask)
    elif mask.ndim == 2:  # 2D case
        total_SA_or_peri = np.sum([region.perimeter for region in regions])

    total_volume = np.sum(volumes) if volumes else 0
    total_convex_volume = np.sum(convex_volumes) if convex_volumes else 0

    return stats, total_volume, total_SA_or_peri, total_convex_volume


def getObjectProperties(labeled_mask, num_fragments, prefix):
    """
    Extracts relevant properties of segmented 3D vessels and returns statistical summaries.

    Parameters
    ----------
    labeled_mask : 3D numpy array
        Labeled segmentation mask where each fragment has a unique ID.
    num_fragments : int
        Number of fragments in the mask.
    prefix : str
        Prefix to be added to the computed statistics (e.g., 'tumor', 'stroma').

    Returns
    -------
    stats : dict
        Dictionary containing statistical summaries of object properties, with prefixed keys.
    """
    # If no fragments are detected, return only the fragment count
    stats = {f"{prefix}_num_fragments": num_fragments}

    if num_fragments == 0:
        return stats  # No objects to process

    # Extract region properties
    regions = regionprops(labeled_mask)

    # Initialize lists for each property
    major_lengths, diameters, volumes, convex_volumes, solidities, elongations = [], [], [], [], [], []

    for region in regions:
        major_lengths.append(region.major_axis_length)
        diameters.append(region.equivalent_diameter_area)
        volumes.append(region.area)
        convex_volumes.append(region.convex_area)
        solidities.append(region.solidity)
        
        # Handle potential errors in minor_axis_length
        try:
            elongations.append(region.major_axis_length / region.minor_axis_length)
        except (ValueError, ZeroDivisionError):
            elongations.append(np.nan)  # Assign NaN if there's an issue

    # Compute statistics for each property
    for name, values in zip(
        ["major_axis_length", "equivalent_diameter_area", "volume", 
         "convex_hull_volume", "elongation", "solidity"],
        [major_lengths, diameters, volumes, convex_volumes, elongations, solidities]
    ):
        values = np.array(values, dtype=np.float64)  # Ensure numerical stability
        stats.update({
            f"{prefix}_{name}_mean": np.nanmean(values),  
            f"{prefix}_{name}_median": np.nanmedian(values),
            f"{prefix}_{name}_std": np.nanstd(values),
            f"{prefix}_{name}_min": np.nanmin(values),
            f"{prefix}_{name}_max": np.nanmax(values),
            f"{prefix}_{name}_sum": np.nansum(values),
        })

    return stats  # Return a dictionary instead of a DataFrame

