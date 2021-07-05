import torch
import numpy as np
import SimpleITK as sitk
import os
from multiprocessing import pool
import pickle
import numpy as np
from skimage.transform import resize
import pandas as pd

def resize_image(image, old_spacing, new_spacing, order=3):
    new_shape = (int(np.round(old_spacing[0]/new_spacing[0]*float(image.shape[0]))),
                 int(np.round(old_spacing[1]/new_spacing[1]*float(image.shape[1]))),
                 int(np.round(old_spacing[2]/new_spacing[2]*float(image.shape[2]))))
    return resize(image, new_shape, order=order, mode='edge')

#def view_patient_raw_data(patient, width=400, height=400):
#    import batchviewer
#    a = []
#    a.append(patient['ed_data'][None])
#    a.append(patient['ed_gt'][None])
#    a.append(patient['es_data'][None])
#    a.append(patient['es_gt'][None])
#    batchviewer.view_batch(np.vstack(a), width, height)

def convert_to_one_hot(seg):
    """Convert segmentation mask into one-hot-encoding

    Args:
        - seg - segmentation mask (n,m)

    Returns:
        - res - segmentation mask in one-hot (k,n,m)
    """
    vals = np.unique(seg)
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for c in range(len(vals)):
        res[c][seg == c] = 1
    return res

def preprocess_image(itk_image, is_seg=False, spacing_target=(1, 0.5, 0.5), keep_z_spacing=False):
    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
    image = sitk.GetArrayFromImage(itk_image).astype(float)
    if keep_z_spacing:
        spacing_target = list(spacing_target)
        spacing_target[0] = spacing[0]
    if not is_seg:
        order_img = 3
        if not keep_z_spacing:
            order_img = 1
        image = resize_image(image, spacing, spacing_target, order=order_img).astype(np.float32)
        image -= image.mean()
        image /= image.std()
    else:
        tmp = convert_to_one_hot(image)
        vals = np.unique(image)
        results = []
        for i in range(len(tmp)):
            results.append(resize_image(tmp[i].astype(float), spacing, spacing_target, 1)[None])
        image = vals[np.vstack(results).argmax(0)]
    return image

def load_concepts_dataset(root_dir="../concepts", info='/media/adri/Adri-687Gb/CONCEPTS_CONTEXT/concepts/superpixels_info_context_with_segments.csv'):#root_dir="/home/people/19203757/unet-training/concepts/"):
    """Loads data into dictionary with the following structure: {<PATIENT NUMBER>_<TYPE>:{pathology:<PATHOLOGY>, segments:<CONCEPT SEGMENTS>}"""
    data = {}
    print(root_dir)
    dirs = os.listdir(root_dir)
    #patient46_patches_15segments_0_path_MINF_ed.npy
#    patients = {int(x.split('_')[0].strip('patient')):x for x in dirs}
#    
#    for i, x in patients.items():
#        a = np.load(os.path.join(root_dir, x), mmap_mode='r', allow_pickle=True)
#        data[i] = {}
#        data[i]['pathology'] = x.split('_')[-2]
#        data[i]['typ'] = x.split('_')[-1].split('.')[0]
#        data[i]['segments'] = a
#
#    patients = {int(x.split('_')[0].strip('patient')):x for x in dirs}
    df = pd.read_csv(info)
    for i in dirs:
        if i.endswith(".npy") and "files" not in i and "context" not in i:
            print(i)
#            print(i.split('_')[0].strip('patient'))
            pat_nb = int(i.split('_')[0].strip('patient'))
#            pat_type = i.split('_')[6].strip('.npy')
#            a = np.load(os.path.join(root_dir, i), mmap_mode='r+', allow_pickle=True)
            a = np.load(os.path.join(root_dir, i), mmap_mode='r+', allow_pickle=True)

            data[str(pat_nb)] = {}
            data[str(pat_nb)]['pathology'] = i.split('_')[-2]
            data[str(pat_nb)]['segments'] = a
            data[str(pat_nb)]['files'] = i

    return data, df



def load_dataset(ids=range(1,101), root_dir="./results/"):
    with open(os.path.join(root_dir, "patient_info.pkl"), 'rb') as f:
        patient_info = pickle.load(f)

    data = {}
    for i in ids:
        if os.path.isfile(os.path.join(root_dir, "pat_%03.0d.npy"%i)):
            a = np.load(os.path.join(root_dir, "pat_%03.0d.npy"%i), mmap_mode='r')
            data[i] = {}
            data[i]['height'] = patient_info[i]['height']
            data[i]['weight'] = patient_info[i]['weight']
            data[i]['pathology'] = patient_info[i]['pathology']
            if np.shape(a)[0] != 2:
                data[i]['ed_data'] = a[0, :]
                data[i]['ed_gt'] = a[1, :]
                data[i]['es_data'] = a[2, :]
                data[i]['es_gt'] = a[3, :]
            else:
                data[i]['ed_data'] = a[0, :]
                data[i]['es_data'] = a[1, :]
    return data

# source -> https://github.com/MIC-DKFZ/ACDC2017
def process_patient(args):
    id, patient_info, folder, folder_out, keep_z_spc = args
    #print id
    # if id in [286, 288]:
    #     return
    patient_folder = os.path.join(folder, "patient%03.0d"%id)
    if not os.path.isdir(patient_folder):
        return
    images = {}

    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d.nii.gz" % (id, patient_info[id]['ed']))
    if os.path.isfile(fname):
        images["ed"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d_gt.nii.gz" % (id, patient_info[id]['ed']))
    if os.path.isfile(fname):
        images["ed_seg"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d.nii.gz" % (id, patient_info[id]['es']))
    if os.path.isfile(fname):
        images["es"] = sitk.ReadImage(fname)
    fname = os.path.join(patient_folder, "patient%03.0d_frame%02.0d_gt.nii.gz" % (id, patient_info[id]['es']))
    if os.path.isfile(fname):
        images["es_seg"] = sitk.ReadImage(fname)

    if "es_seg" in images:
        print (id, images["es_seg"].GetSpacing())

    for k in images.keys():
        #print k
        images[k] = preprocess_image(images[k], is_seg=(k == "ed_seg" or k == "es_seg"),
                                     spacing_target=(10, 1.25, 1.25), keep_z_spacing=keep_z_spc)

    img_as_list = []
    for k in ['ed', 'ed_seg', 'es', 'es_seg']:
        if k not in images.keys():
            print (id, "has missing key:", k)
        else:
            img_as_list.append(images[k][None])
    try:
        all_img = np.vstack(img_as_list)
    except:
        print (id, "has a problem with spacings")
    np.save(os.path.join(folder_out, "pat_%03.0d" % id), all_img.astype(np.float32))


def generate_patient_info(folder, ids_pats):
    patient_info={}
    for id in range(1,ids_pats):
        fldr = os.path.join(folder, 'patient%03.0d'%id)
        if not os.path.isdir(fldr):
            print ("could not find dir of patient ", id)
            continue
        nfo = np.loadtxt(os.path.join(fldr, "Info.cfg"), dtype=str, delimiter=': ')
        
        patient_info[id] = {}
        patient_info[id]['ed'] = int(nfo[0, 1])
        patient_info[id]['es'] = int(nfo[1, 1])
        patient_info[id]['height'] = float(nfo[3, 1])
        patient_info[id]['pathology'] = nfo[2, 1]
        patient_info[id]['weight'] = float(nfo[4, 1])
    return patient_info


def run_preprocessing(folder="./ACDC/training/",ids_pats=1,
                      folder_out = "./results/", keep_z_spacing=True):
    """Process dataset for ACDC challenge
    
    Args:
        - folder - string folder with training data
        - ids_pats - integer number of patients to process 1-100 (ACDC contains 100 patients in training)
        - folder_out - string folder where to store processed data
        - keep_z_spacing - True/False whether to keep z spacing
    
    This functions creates files with preprocessed patients data in folder: folder_out
    """
    ids_pats += 1
    patient_info = generate_patient_info(folder, ids_pats)
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)
    with open(os.path.join(folder_out, "patient_info.pkl"), 'wb') as f:
        pickle.dump(patient_info, f)

    # beware of z spacing!!! see process_patient for more info!
    ids = range(1,ids_pats)
    p = pool.Pool(8)
    p.map(process_patient, zip(ids, [patient_info]*ids_pats, [folder]*ids_pats, [folder_out]*ids_pats, [keep_z_spacing]*ids_pats))
    p.close()
    p.join()

if __name__ == "__main__":
    load_concepts_dataset(root_dir="/media/adri/Adri-687Gb/CARDIAC-TCAV/CONCEPTS/CONCEPTS_OFFICIAL")
    #TODO:
    #run_preprocessing()

