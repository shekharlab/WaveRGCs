import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import bigfish.detection as detection
import bigfish.plot as plot
import bigfish.stack as stack
from cellpose import models
from cellpose import plot as cell_plot
from cellpose import utils

plt.ioff()

def detect_spots(image, spot_radius, physical_scale=None, pixel_size=None, filter_kernel=3, bottom_crop=50, threshold_mod=0):
    scaled_image = image/np.max(image)
    
    filtered_image = stack.remove_background_mean(image, "square", filter_kernel)
    scaled_filtered_image = filtered_image/np.max(filtered_image)

    if (physical_scale != None) and (pixel_size != None):
        raise Exception("Provide either physical_scale (length of scalebar) or pixel_size (length of pixel edge), not both.")
    
    if physical_scale != None:
        pixel_scale = scalebar_length(scaled_image)
        pixel_size = physical_scale/pixel_scale
    elif pixel_size != None:
        pixel_size = pixel_size
    else:
        raise Exception("Must provide either physical_scale (length of scalebar) or pixel_size (length of pixel edge).")

    # spot radius
    spot_radius_px = detection.get_object_radius_pixel(
        voxel_size_nm=(pixel_size, pixel_size), 
        object_radius_nm=(spot_radius, spot_radius),
        ndim=2)
    
    # LoG filter
    rna_log = stack.log_filter(scaled_filtered_image, sigma=spot_radius_px)
    
    # local maximum detection
    mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)
    
    # thresholding
    threshold = threshold_mod*detection.automated_threshold_setting(rna_log, mask)
    spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
        
    return spots, threshold

def mean_spot_intensity(image, spots):
    scaled_image = image/np.max(image)

    spot_intensities = []
    for i in spots:
        spot_intensities.append(scaled_image[i[0], i[1]])

    return np.mean(spot_intensities)

def spots_in_mask(mask, spots, filter_kernel):
    scaled_mask = mask/np.max(mask)

    mask_thresholds = ski.filters.threshold_multiotsu(scaled_mask)
    binary_mask = (scaled_mask > mask_thresholds[1]).astype(np.uint8)
    
    filtered_mask = stack.maximum_filter(binary_mask, "square", filter_kernel)

    remaining_spots = []
    for i in spots:
        if filtered_mask[i[0], i[1]] != 0:
            remaining_spots.append(i)

    return np.array(remaining_spots)

def scalebar_length(image):
    lengths = []
    for i in image:
        row_lengths = []
        current_length = 0
        for j in range(len(i)):
            if i[j] == 1:
                current_length += 1
            else:
                row_lengths.append(current_length)
                current_length = 0

            if j == len(i)-1:
                row_lengths.append(current_length)
        lengths.append(np.max(row_lengths))
    return np.max(lengths)

def label_spots(image, spots, marker_size=10, fig_size=(15,15), saveas="spot_labels.png", show=True):
    fig = plt.figure(figsize=fig_size, frameon=False)
    axs = fig.add_axes([0, 0, 1, 1])
    spots_x = spots[:,1]
    spots_y = spots[:,0]
    axs.axis('off')
    plt.imshow(image, cmap="gray")
    plt.scatter(spots_x, spots_y, facecolors='none', edgecolors='blue', s=marker_size)
    plt.savefig(saveas)
    if show:
        plt.show()
    plt.close()

def segment_cells(images, cyto_rgb, nuc_rgb):
    model = models.CellposeModel(pretrained_model='path/to/cellpose_model/cellpose_new')
    channels = [cyto_rgb, nuc_rgb]
    masks, flows, styles = model.eval(images, diameter=60, channels=channels, flow_threshold=0.6, cellprob_threshold=-1)
    return masks, flows, styles

def segment_cells_no_dapi(images):
    model = models.CellposeModel(pretrained_model='path/to/cellpose/model/cellpose_new')
    masks, flows, styles = model.eval(images, diameter=60, channels=[0,0], flow_threshold=0.6, cellprob_threshold=-1)
    return masks, flows, styles

def count_spots_per_cell(mask, spots):
    spot_counts = []
    for i in range(1,np.max(mask)):
        spot_count = 0
        cell_mask = (mask == i)
        for j in spots:
            if cell_mask[j[0], j[1]]:
                spot_count += 1
        spot_counts.append(spot_count)
    return spot_counts

def show_spots_in_cells(mask, spots, spots_image, marker_size=10, show=True, saveas='spots_in_cells.png'):
    spots_in_cells = []
    for i in range(1,np.max(mask)):
        cell_mask = (mask == i)
        for j in spots:
            if cell_mask[j[0], j[1]]:
                spots_in_cells.append(j)
    
    spots_in_cells = np.array(spots_in_cells)
    
    fig = plt.figure(figsize=(15,15), frameon=False)
    axs = fig.add_axes([0, 0, 1, 1])
    
    axs.axis('off')
    
    if spots_image.shape[-1] < 3 or spots_image.ndim < 3:
        spots_image = cell_plot.image_to_rgb(spots_image)

    outlines = utils.masks_to_outlines(mask)
    outX, outY = np.nonzero(outlines)
    image_outlined = spots_image.copy()
    image_outlined[outX, outY] = np.array([255, 0, 0])  # pure red

    plt.imshow(image_outlined)

    spots_x = spots_in_cells[:,1]
    spots_y = spots_in_cells[:,0]
    plt.scatter(spots_x, spots_y, facecolors='none', edgecolors='blue', s=marker_size)
    
    plt.savefig(saveas)
    if show:
        plt.show()
    plt.close()

def calculate_intensity_per_cell(mask, image):
    mean_intensitites = []
    for i in range(1,np.max(mask)):
        cell_mask = (mask == i)
        cell_intensities = image[cell_mask]
        mean_intensity = np.mean(cell_intensities)
        mean_intensitites.append(mean_intensity)
    return mean_intensitites

def show_segmentation(image, mask, flows, cyto_rgb, nuc_rgb, saveas='segmentation.png', show=True):
    channels = [cyto_rgb, nuc_rgb]
    fig = plt.figure(figsize=(12,5))
    cell_plot.show_segmentation(fig, image, mask, flows[0], channels=channels);
    plt.tight_layout()
    plt.savefig(saveas)
    plt.close()

def show_segmentation_no_dapi(image, mask, flows, saveas='segmentation.png', show=True):
    fig = plt.figure(figsize=(12,5))
    cell_plot.show_segmentation(fig, image, mask, flows[0], channels=[0,0]);
    plt.tight_layout()
    plt.savefig(saveas)
    plt.close()