from mrinet.dataset.preprocessing import load_dataset
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

def create_discovery_data(numSegments_list=30, concepts_ratio=0.2, preview=False):
    """Function that creates discovery dataset for DTCAV method
       it takes equal sample from classes (pathologies) and does SLIC to every 
       data point then crop it and rescale to network input size
       
       Args:
           - numSegments_list -> list that contains sizes of segments 
             that are to be discovered
           - concepts_ratio   -> ratio that defines the amount of 
             samples considered for creating the dataset
             
        Returns:
           - output_data -> dictionary of all concept images found, 
             data - contains images, patient_ids - contains patient ids,
             pathologies contains list with pathologies corresponding to sample
           
        """
    
    if type(numSegments_list) is int:
        numSegments_list = [numSegments_list]
    
    data = load_dataset()
    pathologies = set([data[x]['pathology'] for x in data])
    dcm = [x for x in data if data[x]['pathology'] == 'DCM']
    hcm = [x for x in data if data[x]['pathology'] == 'HCM']
    minf = [x for x in data if data[x]['pathology'] == 'MINF']
    nor = [x for x in data if data[x]['pathology'] == 'NOR']
    rv = [x for x in data if data[x]['pathology'] == 'RV']

    paths = [dcm, hcm, minf, nor, rv]
    types = ["DCM", "HCM", "MINF", "NOR", "RV"]
    sample = []
    sample_type = []
    all_samples = len(data)
    output = []    
    context = []
    #print(np.shape(dcm))
    while len(sample)/all_samples <= concepts_ratio:
        for i, x in enumerate(zip(paths,types)):
            t = x[1]
            x = x[0]
            print("PATHOLOGY: ",t)
            smpl = np.random.choice(x)
            print("PATIENT: ",smpl)
            sample.append(smpl)
            sample_type.append(types[i])
            output_data = []            
            views = []
            for typ in ['ed_data', 'es_data']:
                print("PHASE: ", typ)
                #print(np.shape(data[smpl][typ]))
                X = crop_heart(data[smpl][typ])
                for slice_ in X:
                    if preview:
                        import matplotlib.pyplot as plt
                        plt.imshow(slice_)
                        plt.show()
                    image = img_as_float(slice_.astype(float))
                    # we can do different levels of superpixels
                    for j, numSegments in enumerate(numSegments_list):
                        segments = slic(image, compactness=0.1, n_segments=numSegments, sigma = 5)

                        masks = get_masks(segments)
                        if preview:
                            plt.imshow(segments)
                            plt.show()
                            for seg in masks:
                                plt.imshow(seg)
                                plt.show()
                                plt.imshow(image)
                                plt.show()

                        # probably for every superpixel found
                        for supix in masks:
                            im = mask_image(image,supix)
#                            maski = np.where(segments == 3,segments,0)
                            if preview:
                                plt.imshow(im)
                                plt.show()

                            im = crop_image(im)
                            if preview:
                                plt.imshow(im)
                                plt.show()

                            try:
                                output_data.append(resize_img(im, (348,348)))
                                context.append(mark_boundaries(image, im, color=(255,0,0),mode='thick'))

                            except:
                                continue
            print("NUMBER OF CONCEPTS IMAGES: ",len(output_data))
            np.save('./concepts/patient{}_patches_{}segments_{}_path_{}_{}'.format(sample[-1], numSegments,j,types[i],typ.split('_')[0]), output_data)                      
            np.save('./concepts/context_patient{}_patches_{}segments_{}_path_{}_{}'.format(sample[-1], numSegments,j,types[i],typ.split('_')[0]), context)                      

            output.extend(output_data)

    return {'data':output, 'patients_ids':sample, 'pathologies':sample_type}


if __name__ == "__main__":
    out = create_discovery_data(numSegments_list=15,concepts_ratio=0.3)
#    out = create_discovery_data(numSegments_list=15,concepts_ratio=0.3, preview=True)
