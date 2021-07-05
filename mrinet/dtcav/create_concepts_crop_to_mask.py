from preprocessing import load_dataset
from mrinet.utils.crop_heart import crop_heart
import numpy as np
import pandas as pd
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.transform import resize

def get_masks(tab):
    classes = set(tab.flatten())
    masks = [np.zeros(np.shape(tab))]*len(classes)

    for i, cls in enumerate(classes):
        masks[i] = (tab == cls).astype(int)
    return masks

# https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
def crop_image(img,tol=0.0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
#    print(np.shape(img))
#    print(np.shape(mask))
    return img[np.ix_(mask.any(1),mask.any(0))]

def mask_image(img,mask):
    return np.where(mask, img, mask)

def resize_img(img, new_shape, order=3):
    return resize(img, new_shape, order=order, mode='edge')


import matplotlib.pyplot as plt
def crop_to_mask(data, plot=False):
    """
    
    Handle: 
        - empty masks
        - ed
        - es
    """
    
    cropped = []
    for elem in ['ed', 'es']:
        for i, X in enumerate(zip(data['{}_data'.format(elem)],data['{}_gt'.format(elem)])):
            x = X[0]
            y = X[1]
            # mask and crop
            if sum(y.flatten()) == 0:
                print("ATTENTION: empty mask {}".format(i))
            out = mask_image(x,y)
            out = crop_image(out)
            cropped.append(out)
            if plot:
                plt.imshow(x)
                plt.show()
                plt.imshow(y)
                plt.show()
                plt.imshow(out)
                plt.show()
    return np.array(cropped)

def create_discovery_data(numSegments_list=30, preview=False, approach="mask", crop=False):
    """Function that creates discovery dataset for DTCAV method
       it takes equal sample from classes (pathologies) and does SLIC to every 
       data point then crop it and rescale to network input size
       
       Args:
           - numSegments_list -> list that contains sizes of segments 
             that are to be discovered
           - approach ["mask"|"fourier"] - method to use when cropping the heart
             
        Returns:
           - output_data -> dictionary of all concept images found, 
             data - contains images, patient_ids - contains patient ids,
             pathologies contains list with pathologies corresponding to sample
           
        """

    if type(numSegments_list) is int:
        numSegments_list = [numSegments_list]

    data = load_dataset()  # TODO: LOAD only TRAINING DATA
    output = []
    sample = [] # patients ids
    sample_type = [] # pathologies types
    collect_metadata_only = True
    num_segments = None
    seg_number = -1
    for patient in data:
        t = data[patient]['pathology']
        print("PATHOLOGY: ",t)
        print("PATIENT: ", patient)
        sample.append(patient)
        sample_type.append(t)
        output_data = []
        name_of_files = []
        views = []
        context = []

        for typ in ['ed_data', 'es_data']:
            print("PHASE: ", typ)
            try:
                if approach == "fourier":
                    X = crop_heart(data[patient][typ])
                else:
                    X = crop_to_mask(data[patient])
                for slice_num, slice_ in enumerate(X):
                    if preview:
                        import matplotlib.pyplot as plt
                        plt.imshow(slice_)
                        plt.show()
                    image = img_as_float(slice_.astype(float))
                    # we can do different levels of superpixels
                    for j, numSegments in enumerate(numSegments_list):
                        segments = slic(image, compactness=0.1, n_segments=numSegments, sigma = 5)
                        context.append(segments)
                        masks = get_masks(segments)
                        #print(np.shape(masks))
    #                    name_of_files.append("pathology_{}_patient_{}_typ_{}_slice_{}_segments{}".format(t,smpl,typ,slice_num,len(masks)))
                        #print(len(name_of_files))
    
                        if preview:
                            plt.imshow(segments)
                            plt.show()
                            for seg in masks:
                                plt.imshow(seg)
                                plt.show()
                                plt.imshow(image)
                                plt.show()
    
                        out = []
                        f_names = []
                        # probably for every superpixel found
                        for pix_j, supix in enumerate(masks):
                            imi = mask_image(image,supix)
                            if preview:
                                plt.imshow(imi)
                                plt.show()
    
                            if crop:
                                imi = crop_image(imi)
                                if preview:
                                    plt.imshow(imi)
                                    plt.show()
    
                            try:                                
                                out.append(resize_img(imi, (348, 348)))
                                name_of_files.append("pathology_{}_patient_{}_typ_{}_slice_{}_segments{}".format(t,patient,typ,slice_num,pix_j))
    
                            except:
                                print(np.shape(imi))                                
                                print("ERROR")
                                continue
    
                        output_data.extend(out)
                        name_of_files.extend(f_names)
                        #context.append(mark_boundaries(image, imi, color=(255,0,0),mode='thick'))
                        #context.append(imi)
                        num_segments = numSegments
                        seg_number = j

            except Exception as e:
                print(e)
                print("Can't process patient: {}, phase: {}, skipping...".format(sample[-1],typ))
                continue
        print("NUMBER OF CONCEPTS IMAGES: ",len(output_data))
        np.save('./concepts/patient{}_patches_{}segments_{}_path_{}_{}'.format(sample[-1], seg_number,num_segments,t,typ.split('_')[0]), output_data)
        np.save('./concepts/files_patient{}_patches_{}segments_{}_path_{}_{}'.format(sample[-1], seg_number,num_segments,t,typ.split('_')[0]), name_of_files)
        np.save('./concepts/context_patient{}_patches_{}segments_{}_path_{}_{}'.format(sample[-1],seg_number,num_segments,t,typ.split('_')[0]), context)
        output.extend(output_data)
        
        #break
    return {'data':output, 'patients_ids':sample, 'pathologies':sample_type}

if __name__ == "__main__":
    _ = create_discovery_data(numSegments_list=5, preview=False)
