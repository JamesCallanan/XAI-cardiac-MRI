from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform
from batchgenerators.transforms import MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform
from batchgenerators.augmentations.utils import resize_image_by_padding, random_crop_3D_image_batched, \
    random_crop_2D_image_batched, center_crop_3D_image_batched, center_crop_2D_image_batched
from batchgenerators.dataloading import DataLoaderBase
import numpy as np
import os
import pandas as pd
from mrinet.dataset.preprocessing import load_dataset, load_concepts_dataset
import torch

try:
    import nibabel as nib
except:
    pass


def check_equal(lst1, lst2, v=0):
    cnt = 0
    for x,y in zip(lst1,lst2):
        if x != y:
            if v:
                print("{} -> {}".format(x,y))
            cnt += 1
    return cnt

# source -> https://github.com/MIC-DKFZ/ACDC2017
class BatchGenerator(DataLoaderBase):
    """Generator of batches, inherits after DataLoaderBase
    
    Attributes
    ----------
    PATCH_SIZE : tuple(int,int) - size of the image to be returned
    phase : str - parameter to distinguish which batch function to use to return batches (train, metrics, concepts)
    
    Methods
    -------
    generate_train_batch() - method required by the parent class DataLoaderBase, 
                             here serves as a dispatcher to different scenarios of batch generation.
    train_batch() - generate batch for the training phase, it is randomized
    concepts_batch() - generate batch of concepts images in DTCAV method, diffrence from the previous is in what is returned
    ordered_batch() - generate batch in the order of patients"""
    def __init__(self, data, BATCH_SIZE, phase="train", patch_size=(256, 256), num_batches=None, seed=False):
        self.PATCH_SIZE = patch_size
        self.phase = phase
        DataLoaderBase.__init__(self, data, BATCH_SIZE, num_batches=num_batches, seed=seed)

    def generate_train_batch(self):
        """Dispatcher of which batch generation scenario to use
        
        scenarios:
            - metrics - generate batches in deterined order of patients (used for acdc_metrics script)
            - train - random batches from different patients, used in training and validation sets
            - concepts - used to generate batches of concept images, the return form is different hence the new function"""
        if self.phase == "metrics":
            return self.ordered_batch()
        elif self.phase == "train":
            return self.train_batch()
        elif self.phase == "concepts":
            return self.concepts_batch()
        elif self.phase == "eval":
            return self.eval_batch()


    def train_batch(self):
        """Scenario to generate batches randomly, used to do batch generator for training and validation sets"""
        data = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
        seg = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
        types = np.random.choice(['ed', 'es'], self.BATCH_SIZE, True)
        patients = np.random.choice(list(self._data.keys()), self.BATCH_SIZE, True)
        pathologies = []
        for nb in range(self.BATCH_SIZE):
            shp = self._data[patients[nb]][types[nb]+'_data'].shape
            slice_id = np.random.choice(shp[0])
            tmp_data = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_data'][slice_id], (max(shp[1],
                                                self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)
            tmp_seg = resize_image_by_padding(self._data[patients[nb]][types[nb]+'_gt'][slice_id], (max(shp[1],
                                                self.PATCH_SIZE[0]), max(shp[2], self.PATCH_SIZE[1])), pad_value=0)

            # not the most efficient way but whatever...
            tmp = np.zeros((1, 2, tmp_data.shape[0], tmp_data.shape[1]))
            tmp[0, 0] = tmp_data
            tmp[0, 1] = tmp_seg
            tmp = random_crop_2D_image_batched(tmp, self.PATCH_SIZE)
            data[nb, 0] = tmp[0, 0]
            seg[nb, 0] = tmp[0, 1]
            pathologies.append(self._data[patients[nb]]['pathology'])
        return {'data':data, 'seg':seg, 'types':types, 'patient_ids': patients, 'pathologies':pathologies}

    def concepts_batch(self):
        """Scenario to generate batches of concept images, the return form is different from training phase hence the new function."""
        patients_ids = []
        pathologies = []
        segments = []
        types = []
        files = []
        data, info = self._data


        original_ind = info.index
        info['files_cnt'] = info.groupby(['files']).cumcount().astype(str)
        info['files'] = info[['files', 'files_cnt']].apply(lambda x: ''.join(x), axis=1)
        del info['files_cnt']
        
        print(info.head())
        for i, x in data.items():
            #print(i)
            #patients_ids.extend([int(i.split('_')[0])]*len(x['segments']))
            #pathologies.extend([x['pathology']]*len(x['segments']))
            #types.extend([i.split('_')[1]]*len(x['segments']))
            segments.extend(x['segments'])
            patients_ids.extend([i]*len(x['segments']))
            files.extend([x['files']+str(jj) for jj in range(len(x['segments']))])
        df = pd.DataFrame()
        print("segments: ",len(segments))
        df["segments_data"] = segments
        print("files: ",len(files))
        print(files[:5])
        df["files"] = files
        print(len(df))
        xxx = check_equal(info['patients'], patients_ids, v=0)
        if xxx != 0:         #TODO check if data in info and in data is aligned
            print("ERROR: Patients numbers are not equal between the two sources, number of misaligned: {}, doing join...".format(xxx))
            zz = info.join(df.set_index('files'), lsuffix='_caller', rsuffix='_other', on="files")
            print(len(zz),len(info),len(segments))
            print(zz.head())
            is_NaN = zz.isnull() 
            row_has_NaN = is_NaN.any(axis=1)
            rows_with_NaN = zz[row_has_NaN]
            print(rows_with_NaN)
            zz.dropna(inplace=True)

        else:
            print("Patient numbers ALIGNED")
        if len(segments) == len(info):
            print("The sizes of metadata and actual data are equal")
        else:
            print("ERROR ITS NOT EQUAL CHECKKKK")
            print("segments: ",len(segments), " info: ",len(info))


        return {'segments': zz['segments_data'], 'patient_ids': zz['patients'], 'pathologies':zz['pathologies'], 'types':zz['types'], 'slices':zz['slices'], 'files':zz['files'], 'seg_num': zz['segments']}

    def ordered_batch(self):    
        """Scenario to generate batches in order,(used for acdc_metrics script)."""
        patients = []
        pathologies = []
        data_es_all = []
        seg_es_all = []
        data_ed_all = []
        seg_ed_all = []        
        for patient_nb in range(1,len(self._data)+1):
            patients.append(patient_nb)    
            data_es = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            seg_es = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            data_ed = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            seg_ed = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)

            for nb in range(self.BATCH_SIZE):
                if 'es_data' in self._data[patient_nb]:                
                    shp_es = self._data[patient_nb]['es_data'].shape
                    for elem in range(shp_es[0]):
                        tmp_data_es = resize_image_by_padding(self._data[patient_nb]['es_data'][elem], (max(shp_es[1],
                                                            self.PATCH_SIZE[0]), max(shp_es[2], self.PATCH_SIZE[1])), pad_value=0)
                        tmp_seg_es = resize_image_by_padding(self._data[patient_nb]['es_gt'][elem], (max(shp_es[1],
                                                            self.PATCH_SIZE[0]), max(shp_es[2], self.PATCH_SIZE[1])), pad_value=0)
                else:
                    tmp_data_es = []
                    tmp_seg_es = []
                    
                if 'ed_data' in self._data[patient_nb]:                    
                    shp_ed = self._data[patient_nb]['ed_data'].shape
                    for elem in range(shp_ed[0]):
                        tmp_data_ed = resize_image_by_padding(self._data[patient_nb]['ed_data'][elem], (max(shp_ed[1],
                                                            self.PATCH_SIZE[0]), max(shp_ed[2], self.PATCH_SIZE[1])), pad_value=0)
                        tmp_seg_ed = resize_image_by_padding(self._data[patient_nb]['ed_gt'][elem], (max(shp_ed[1],
                                                            self.PATCH_SIZE[0]), max(shp_ed[2], self.PATCH_SIZE[1])), pad_value=0)
                else:
                    tmp_data_ed = [] 
                    tmp_seg_ed = []
                    
                # not the most efficient way but whatever...
                tmp = np.zeros((1, 2, tmp_data_es.shape[0], tmp_data_es.shape[1]))
                tmp[0, 0] = tmp_data_es
                tmp[0, 1] = tmp_seg_es
                tmp = center_crop_2D_image_batched(tmp, self.PATCH_SIZE)
                data_es[nb, 0] = tmp[0, 0]
                seg_es[nb, 0] = tmp[0, 1]

                tmp = np.zeros((1, 2, tmp_data_ed.shape[0], tmp_data_ed.shape[1]))
                tmp[0, 0] = tmp_data_ed
                tmp[0, 1] = tmp_seg_ed
                tmp = center_crop_2D_image_batched(tmp, self.PATCH_SIZE)
                data_ed[nb, 0] = tmp[0, 0]
                seg_ed[nb, 0] = tmp[0, 1]

                pathologies.append(self._data[patient_nb]['pathology'])
                data_es_all.append(data_es)
                seg_es_all.append(seg_es)                
                data_ed_all.append(data_ed)          
                seg_ed_all.append(seg_ed)                            

        return {'es_data':data_es_all, 'es_seg':seg_es_all, 'ed_data':data_ed_all, 'ed_seg':seg_ed_all, 'patient_ids': patients, 'pathologies':pathologies}


    def eval_batch(self):
        self.BATCH_SIZE = 1
        patients = []
        frames = []
        pathologies = []
        data_es_all = []
        seg_es_all = []
        data_ed_all = []
        seg_ed_all = []        
        for patient_nb in range(1,len(self._data)+1):   
            data_es = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            seg_es = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            data_ed = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)
            seg_ed = np.zeros((self.BATCH_SIZE, 1, self.PATCH_SIZE[0], self.PATCH_SIZE[1]), dtype=np.float32)

            for nb in range(self.BATCH_SIZE):
                tmp_data_es = []
                tmp_seg_es = []
                tmp_data_ed = [] 
                tmp_seg_ed = []
                
                if 'es_data' in self._data[patient_nb]:                
                    shp_es = self._data[patient_nb]['es_data'].shape
                    for elem in range(shp_es[0]):
                        tmp_data_es.append(resize_image_by_padding(self._data[patient_nb]['es_data'][elem], (max(shp_es[1],
                                                            self.PATCH_SIZE[0]), max(shp_es[2], self.PATCH_SIZE[1])), pad_value=0))
                        tmp_seg_es.append(resize_image_by_padding(self._data[patient_nb]['es_gt'][elem], (max(shp_es[1],
                                                            self.PATCH_SIZE[0]), max(shp_es[2], self.PATCH_SIZE[1])), pad_value=0))
                    
                if 'ed_data' in self._data[patient_nb]:                    
                    shp_ed = self._data[patient_nb]['ed_data'].shape
                    for elem in range(shp_ed[0]):
                        tmp_data_ed.append(resize_image_by_padding(self._data[patient_nb]['ed_data'][elem], (max(shp_ed[1],
                                                            self.PATCH_SIZE[0]), max(shp_ed[2], self.PATCH_SIZE[1])), pad_value=0))
                        tmp_seg_ed.append(resize_image_by_padding(self._data[patient_nb]['ed_gt'][elem], (max(shp_ed[1],
                                                            self.PATCH_SIZE[0]), max(shp_ed[2], self.PATCH_SIZE[1])), pad_value=0))
                    
                # not the most efficient way but whatever...
                
                es_sample = np.array(tmp_data_es)
                es_seg_sample = np.array(tmp_seg_es)
                ed_sample = np.array(tmp_data_ed)
                ed_seg_sample = np.array(tmp_seg_ed)     
                
                for tmp_data_es, tmp_seg_es in zip(es_sample, es_seg_sample):
                    #print(tmp_data_es.shape)
                    tmp = np.zeros((1, 2, tmp_data_es.shape[0], tmp_data_es.shape[1]))
                    tmp[0, 0] = tmp_data_es
                    tmp[0, 1] = tmp_seg_es
                    tmp = center_crop_2D_image_batched(tmp, self.PATCH_SIZE)
                    data_es[nb, 0] = tmp[0, 0]
                    seg_es[nb, 0] = tmp[0, 1]
                    data_es_all.append(data_es)
                    seg_es_all.append(seg_es)                

                for tmp_data_ed, tmp_seg_ed in zip(ed_sample, ed_seg_sample):                    
                    tmp = np.zeros((1, 2, tmp_data_ed.shape[0], tmp_data_ed.shape[1]))
                    tmp[0, 0] = tmp_data_ed
                    tmp[0, 1] = tmp_seg_ed
                    tmp = center_crop_2D_image_batched(tmp, self.PATCH_SIZE)
                    data_ed[nb, 0] = tmp[0, 0]
                    seg_ed[nb, 0] = tmp[0, 1]
                    data_ed_all.append(data_ed)          
                    seg_ed_all.append(seg_ed)     
                maxi = max(len(ed_seg_sample), len(es_seg_sample))
                for zz in range(maxi):
                    pathologies.append(self._data[patient_nb]['pathology'])
                    patients.append(patient_nb)
                    frames.append(zz)
                
        return {'es_data':data_es_all, 'es_seg':seg_es_all, 'ed_data':data_ed_all, 'ed_seg':seg_ed_all, 'patient_ids': patients, 'pathologies':pathologies, 'frame_ids':frames}


def create_data_gen(patient_data_train, batch_size, patch_size, num_batches, num_classes, phase="train", transforms=False,
                                  num_workers=1, num_cached_per_worker=2,
                                  do_elastic_transform=False, alpha=(0., 1300.), sigma=(10., 13.),
                                  do_rotation=False, a_x=(0., 2*np.pi), a_y=(0., 2*np.pi), a_z=(0., 2*np.pi),
                                  do_scale=True, scale_range=(0.75, 1.25), seeds=None):
    """Creates data generator
    
    Args:
        - patient_data_train
        - batch_size
        - patch_size
        - num_batches
        - num_classes
        - phase - what batch generation scenario to use (train, concepts, metrics)
        - transforms - whether to perform transformations or not
        - num_workers
        - num_cached_per_worker
        - do_elastic_transform
        - alpha
        - sigma
        - do_rotation
        - a_x
        - a_y
        - a_z
        - do_scale
        - scale_range
        - seeds
        
    Returns:
        tr_mt_gen - generator with data"""
    if seeds is None:
        seeds = [None]*num_workers
    elif seeds == 'range':
        seeds = range(num_workers)
    else:
        assert len(seeds) == num_workers
        
    data_gen = BatchGenerator(patient_data_train, batch_size, phase, num_batches=num_batches, seed=False,
                                       patch_size=patch_size)
       
    if transforms:
        tr_transforms = []
        tr_transforms.append(MirrorTransform((0, 1)))
        tr_transforms.append(RndTransform(SpatialTransform(patch_size, list(np.array(patch_size)//2),
                                                           do_elastic_transform, alpha,
                                                           sigma,
                                                           do_rotation, a_x, a_y,
                                                           a_z,
                                                           do_scale, scale_range, 'constant', 0, 3, 'constant',
                                                           0, 0,
                                                           random_crop=False), prob=0.67,
                                          alternative_transform=RandomCropTransform(patch_size)))
        tr_transforms.append(ConvertSegToOnehotTransform(range(num_classes), seg_channel=0, output_key='seg_onehot'))    
        tr_composed = Compose(tr_transforms)
    
    #     tr_mt_gen = MultiThreadedAugmenter(data_gen_train, tr_composed, num_workers, num_cached_per_worker, seeds)
    #     tr_mt_gen.restart()
    else:
        tr_composed = Compose([ConvertSegToOnehotTransform(range(num_classes), seg_channel=0, output_key='seg_onehot')])

    tr_mt_gen = SingleThreadedAugmenter(data_gen, tr_composed) 

    return tr_mt_gen



def save_nii(output_folder, patient_id, frame_suffix, data):
    """Wrapper function to save data from the patient into nii format.

    Args:
        - output_folder - folder to store the results (e.g. ./submission)
        - patient_id - id of patient
        - frame_suffix - _ED or _ES, distinguish between end-diastole and end-systole
        - data - image to be saved

    Function creates file of format patient1_ED.nii.gz into given folder e.g. ./submission

    Example:
    >>>output_folder = './submission'
    >>># Save submission file for patient 1 end-diastolic frame: patient1_ED.nii.gz 

    >>>frame_suffix = '_ED'
    >>>patient_id = '1'

    >>>save_nii(output_folder, patient_id, frame_suffix, dat['ed_data'][1])
    """
    
    
    image_file_name = os.path.join(output_folder,
                            'patient' + patient_id + '_' + frame_suffix + '.nii.gz')
    print('saving to: %s' % image_file_name)

    nimg = nib.Nifti1Image(data, affine=None)
    nimg.to_filename(image_file_name)

if __name__ == "__main__":
    train_data, info = load_concepts_dataset(root_dir="/media/adri/Adri-687Gb/CONCEPTS_CONTEXT_PIX_NUM/concepts/data/")
    model_path = "/media/adri/Adri-687Gb/Studies/PhD_UCD/notebooks/temp/intermediate_model_11th_epoch_175.pth"
    NUM_BATCHES = 100
    model = torch.load(model_path, map_location='cpu')
    data_gen_training = create_data_gen((train_data,info), 10, 348, NUM_BATCHES, 4, "concepts")
    D = next(data_gen_training)
    print(D.keys())

