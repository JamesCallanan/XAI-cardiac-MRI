from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing, disk
import numpy as np

def thresh_segmentation(patient_img):
    """Returns matrix
    Segmententation of patient_img with k-means
    """
    #Z = np.float32(np.ravel(patient_img))
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #flags = cv2.KMEANS_RANDOM_CENTERS
    #compactness, labels, centers = cv2.kmeans(Z, 2, None, criteria, 10, flags)
    #center = np.uint8(centers)
    thresh = threshold_otsu(patient_img)
    binary = patient_img > thresh
    return binary

def segment_multiple(patient_img):
    """Returns list
    List of segmented slices with function thresh_segmentation()
    """
    num_slices, height, width = patient_img.shape
    segmented_slices = np.zeros((num_slices, height, width))

    for i in range(num_slices):
        seg_slice = thresh_segmentation(patient_img[i])
        if seg_slice.sum() > seg_slice.size * 0.5:
            seg_slice = 1 - seg_slice
        segmented_slices[i] = seg_slice

    return segmented_slices

# Based on https://gist.github.com/ajsander/fb2350535c737443c4e0#file-tutorial-md
def fourier_time_transform_slice(image_3d):
    '''
    3D array -> 2D array
    [slice, height, width] -> [height, width]
    Returns (width, height) matrix
    Fourier transform for 3d data (time,height,weight)
    '''
    # Apply FFT to the selected slice
    fft_img_2d = np.fft.fftn(image_3d)[:, :]
    return np.abs(np.fft.ifftn(fft_img_2d))

def fourier_time_transform(patient_images):
    '''
    4D array -> 3D array (compresses time dimension)
    Concretely, [slice, time, height, width] -> [slice, height, width]
    Description: Fourier transform for analyzing movement over time.
    '''

    ftt_image = np.array([
        fourier_time_transform_slice(patient_slice)
        for patient_slice in patient_images
    ])
    return ftt_image

def roi_mean_yx(patient_img):
    """Returns mean(y) and mean(x) [double]
    Mean coordinates in segmented patients slices.
    This function performs erosion to get a better result.
    Original: See https://nbviewer.jupyter.org/github/kmader/Quantitative-Big-Imaging-2019/blob/master/Lectures/06-ShapeAnalysis.ipynb
    """
    seg_slices = segment_multiple(patient_img)
    num_slices = seg_slices.shape[0]
    y_all, x_all = np.zeros(num_slices), np.zeros(num_slices)
    neighborhood = disk(2)
    
    for i,seg_slice in enumerate(seg_slices):
        # Perform erosion to get rid of wrongly segmented small parts
        seg_slices_eroded = binary_erosion(seg_slice, neighborhood) 
        
        # Filter out background of slice, after erosion [background=0, foreground=1]
        y_coord, x_coord = seg_slices_eroded.nonzero()
        
        # Save mean coordinates of foreground 
        y_all[i], x_all[i] = np.mean(y_coord), np.mean(x_coord)
    
    # Return mean of mean foregrounds - this gives an estimate of ROI coords.
    mean_y = int(np.mean(y_all))
    mean_x = int(np.mean(x_all))
    return mean_y, mean_x

def crop_roi(img, dim_y, dim_x, cy, cx):
    """
    Crops an image from the given coords (cy, cx), such that the resulting img is of
    dimensions [dim_y, dim_x], i.e. height and width.
    Resulting image is filled out from top-left corner, and remaining pixels are left black.
    """
    cy, cx = int(round(cy)), int(round(cx))
    h, w = img.shape
    if dim_x > w or dim_y > h: raise ValueError('Crop dimensions larger than image dimension!')
    new_img = np.zeros((dim_y, dim_x))
    dx, dy = int(dim_x / 2), int(dim_y / 2)
    dx_odd, dy_odd = int(dim_x % 2 == 1), int(dim_y % 2 == 1)

    # Find boundaries for cropping [original img]
    dx_left = max(0, cx - dx)
    dx_right = min(w, cx + dx + dx_odd)
    dy_up = max(0, cy - dy)
    dy_down = min(h, cy + dy + dy_odd)

    # Find how many pixels to fill out in new image
    range_x = dx_right - dx_left
    range_y = dy_down - dy_up
    

    # Fill out new image from top left corner
    # Leave pixels outside range as 0's (black)
    new_img[0:range_y, 0:range_x] = img[dy_up:dy_down, dx_left:dx_right]
    return new_img


def crop_heart(images_4d, heart_pixel_size=200):
    # Find center for cropping
    ft_imges = fourier_time_transform(images_4d)
    y, x = roi_mean_yx(ft_imges)
    
    # Create new 4d image array
    num_slices, h, w = images_4d.shape
    heart_cropped_img_4d = np.zeros((num_slices, heart_pixel_size, heart_pixel_size))
    
    for i in range(num_slices):
        heart_cropped_img_4d[i] = crop_roi(images_4d[i], heart_pixel_size, heart_pixel_size, y, x)
    
    #plot_patient_data_4d(heart_cropped_img_4d) # plot cropped heart
    return heart_cropped_img_4d
