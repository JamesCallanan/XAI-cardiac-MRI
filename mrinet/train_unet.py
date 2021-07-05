import os
import time
import copy
import json
import torch
from datetime import datetime
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose, RndTransform
from batchgenerators.transforms import SpatialTransform
from batchgenerators.transforms import MirrorTransform
from batchgenerators.transforms import GammaTransform, ConvertSegToOnehotTransform
from batchgenerators.transforms import RandomCropTransform
from batchgenerators.augmentations.utils import resize_image_by_padding, random_crop_3D_image_batched, \
    random_crop_2D_image_batched, center_crop_3D_image_batched, center_crop_2D_image_batched
from batchgenerators.dataloading import DataLoaderBase

from mrinet.dataset.preprocessing import load_dataset
from mrinet.models.unet import Unet
from mrinet.models.fcn import FCN8s
from mrinet.loss import dice_loss
from mrinet.dataset.data_loaders import BatchGenerator, create_data_gen



# source -> https://github.com/usuyama/pytorch-unet
def calc_loss(pred, target, metrics, bce_weight=0.5):
    """Calculates loss

    Args:
        - pred - predictions
        - target - ground truth
        - metrics - dictionary with on-going metrics (bce, dice, loss)
        - bce_weight [0.5] - weight of the binary cross entropy as a component of a loss, if 0 no bce included only DICE score is used

    Returns:
        - loss
    
    This function modifies metrics dictionary."""
    bce = F.binary_cross_entropy_with_logits(pred, target)
        
    pred = F.sigmoid(pred)
    
    dice = dice_loss(pred, target)
    
    loss = bce * bce_weight + dice * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

# adapted from -> https://github.com/usuyama/pytorch-unet
def print_metrics(metrics, epoch_samples, phase):    
    """Print metrics normalized to number of samples in the epoch

    Args:
        - metrics - dictionary with on-going metrics (bce, dice, loss)
        - epoch_samples - number of samples in the epoch
        - phase - train or val, phase for which metrics are reported

    Returns:
        - metrics_values - metrics normalized to number of samples (dictionary)

    Prints values of metrics."""
    outputs = []
    metrics_values = {}
    for k in metrics.keys():
        if k == "dice":
            outputs.append("{}: {:4f}".format(k,1 - metrics[k] / epoch_samples))
        else:
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        metrics_values[k] = metrics[k] / epoch_samples

        
    print("{}: {}".format(phase, ", ".join(outputs))) 
    return metrics_values


# adapted from -> https://github.com/usuyama/pytorch-unet    
def train_model(model, optimizer, scheduler, dataloaders, cf):
    """Trains the model
    
    Args:
        - model - model to be trained
        - optimizer - e.g. Adam
        - scheduler - decides on how to update learning rate
        - dataloaders - dictionary of dataloaders for two phases: train and val e.g. {"train":DATALOADER2,"val":DATALOADER2}
        - cf - configurations, all the parameters needed, e.g. number of epochs etc..
        
    Returns:
        - model - trained model
        - train_history - list of all losses from the training set across epochs
        - val_history - list of all losses from the validation set across epochs
        
    This function creates several artefacts:
	    intermediate_model_epoch_X.pth - models trained so far (use del_models from utils to clean the space)
            intermed_val_history.csv - save history of losses before the training ends
            intermed_train_history.csv - same as above but for val losses
            intermed_metrics.txt - reports metrics dice, bce, loss
        """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_history = []
    val_history = []
    for epoch in range(cf['n_epochs']):
        print('Epoch {}/{}'.format(epoch, cf["n_epochs"] - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            print(phase) 
            for i, data_dict in enumerate(dataloaders[phase]):
                
                inputs = torch.from_numpy(data_dict['data']).to(device)
                types = data_dict['types']
                patient_ids = data_dict['patient_ids']
                pathologies = data_dict['pathologies']
                seg_onehot = data_dict['seg_onehot']
                labels = torch.from_numpy(data_dict['seg_onehot']).to(device) # we need an output encoded into 4 layers          

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels[:,:,92:cf["INPUT_PATCH_SIZE"]-92,92:cf["INPUT_PATCH_SIZE"]-92]
                    loss = calc_loss(outputs, labels, metrics, bce_weight=cf["BCE_WEIGHT"])
                    if i % 250 == 0:
                        print(loss)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            metrics_values = print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            if phase == 'train':
                train_history.append(epoch_loss)
            elif phase == 'val':
                val_history.append(epoch_loss)            

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        if epoch % 5 == 0:
            torch.save(model, './intermediate_model_epoch_{}.pth'.format(epoch)) 
            header = 'loss history'
            np.savetxt("./intermed_val_history.csv", val_history, delimiter=",", fmt='%s', header=header)
            np.savetxt("./intermed_train_history.csv", train_history, delimiter=",", fmt='%s', header=header)
            with open("./intermed_metrics.txt", 'w') as f:
                f.write(str([metrics[m]/epoch_samples for m in metrics]) + " phase: {}".format(phase))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    with open("./final_metrics.json", 'w') as f:
        f.write(json.dumps({m: metrics[m]/epoch_samples for m in metrics}))

    return model, train_history, val_history


# source -> https://github.com/MIC-DKFZ/ACDC2017
def get_split(fold, ids=101, seed=12345):
    """Splits patients into train and test sets.
    
    Args:
        - fold - how many parts in the original dataset
        - how many indexes in the dataset
        - seed - the results are going to be the same each time
        
    Returns:
        - train_keys - train indexes
        - test_keys - test indexes"""
    all_keys = np.array(ids).reshape(-1,1)  #np.arange(1, ids)
    if len(ids) > 5:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        splits = kf.split(all_keys)
        for i, (train_idx, test_idx) in enumerate(splits):
            train_keys = [x[0] for x in all_keys[train_idx]]
            test_keys = [x[0] for x in all_keys[test_idx]]
            if i == fold:
                break
    else:
        train_keys = all_keys
        test_keys = []
    return train_keys, test_keys

def register_experiment(params, n_samples, opts={}, path="experiments-test.json", model_str="fcn_opts"):
    """Register experiment, two phases, training and evaluation
    
    fields: id, timestamp, major training parameters, dataset parameters, dice result"""
    
    with open(params) as f:
        params = json.loads(f.read())
    
    if os.path.exists(path):
        with open(path) as f:
            data = json.loads(f.read())
    else:
        data = {}
        
    if data:
        key = int(max(data.keys(), key=lambda x: int(x))) + 1
    else:
        key = 1
    tmp = {'id': key}
    tmp["timestamp"] = datetime.strftime(datetime.now(), "%d-%m-%y_%H-%M")
    tmp["BATCH_SIZE"] = params["BATCH_SIZE"]
    tmp["INPUT_PATCH_SIZE"] = params["INPUT_PATCH_SIZE"]
    tmp["num_classes"] = params["model_opts"][model_str]["n_classes"]
    tmp["EXPERIMENT_NAME"] = params["EXPERIMENT_NAME"]
    tmp["results_dir"] = params["results_dir"]
    tmp["n_epochs"] = params["n_epochs"]
    tmp["lr_decay"] = params["lr_decay"]
    tmp["base_lr"] = params["base_lr"]
    tmp["n_batches_per_epoch"] = params["n_batches_per_epoch"]
    tmp["weight_decay"] = params["weight_decay"]
    if "unet" in model_str:
        tmp["n_input_channels"] = params["model_opts"][model_str]["n_input_channels"]
        tmp["NUMBER_OF_FILTERS"] = params["model_opts"][model_str]["NUMBER_OF_FILTERS"]
    tmp["BCE_WEIGHT"] = params["BCE_WEIGHT"]

    tmp.update(opts)
    tmp["data samples"] = n_samples
    data[key] = tmp
    
    with open(path, 'w') as f:
        f.write(json.dumps(data))

    return key
    
    
def update_experiments_registry(key, metrics, path="experiments-test.json"):
    """Update registry with results of training"""
    if os.path.exists(path):
        with open(path) as f:
            data = json.loads(f.read())
    else:
        print("ERROR: Experiment was not registered in the first place! Quitting")
        return

    data[key]["results"] = metrics
    if "dice" in metrics:
        data[key]["dice"] = round(1 - metrics['dice'], 4)
    if "bce" in metrics:
        data[key]["bce"] = round(metrics['bce'], 4)
    if "loss" in metrics:
        data[key]["loss"] = round(metrics['loss'], 4)
    
    with open(path, 'w') as f:
        f.write(json.dumps(data))    
        

def setup_training(path, ids, root_dir, result_dir, register="experiments.json", opts={}, model_str='fcn_opts'):
    """Set-up environment to train the model, all the parameters, and history collection
       creates necessary folders, setup optimizer, model, test, val and train data.

    Args: 
        - path - file with configuration parameters (params.json)
        - ids - number of patients to use in training
        
    Returns:
        -"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opts = {"Optimizer": "DEFAULT-Adam",
    "BatchNorm": "DEFAULT-True",
    "Activation Fcn": "DEFAULT-LeakyRelu",
    "Dropout": "DEFAULT-False",
    "Transformations": {"Mirror":"DEFAULT-False", "Rotation":"DEFAULT-False", "Scale":"DEFAULT-False", "Elastic":"DEFAULT-False"},
    "Post-processing": "DEFAULT-False",
    "Spatial resolution": "DEFAULT-1.25 x 1.25 x Zorg"}

    experiment_key = register_experiment(path, ids, opts, register, model_str) 
    with open(path) as f:
        tmp = f.read() 
        cf = json.loads(tmp)
    print("Training with the following parameters:")
    print("".join(["{}:{}\n".format(x,cf[x]) for x in cf]))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    with open(result_dir+"/params.json", "w") as f:
        f.write(json.dumps(cf))
     
    BATCH_SIZE = cf['BATCH_SIZE']
    INPUT_PATCH_SIZE = (cf['INPUT_PATCH_SIZE'],cf['INPUT_PATCH_SIZE'])
    num_classes = cf['model_opts'][model_str]['n_classes']
    EXPERIMENT_NAME = cf['EXPERIMENT_NAME']
    results_dir = cf['results_dir']
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    n_epochs = cf['n_epochs']
    lr_decay = np.float32(cf['lr_decay'])
    base_lr = np.float32(cf['base_lr'])
    n_batches_per_epoch = cf['n_batches_per_epoch']
    num_class = cf['model_opts'][model_str]['n_classes']

#    opts = {'duke_unet_opts':{"n_input_channels":cf['n_input_channels'],"n_classes":cf['num_classes'],"NUMBER_OF_FILTERS":cf["NUMBER_OF_FILTERS"]}}
    opts = cf['model_opts']
    if "unet" in model_str:
        model = Unet(opts).to(device)
    elif "fcn" in model_str:
        model = FCN8s(opts).to(device)
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model.parameters(), lr=cf['base_lr'])
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=cf['lr_decay'])
    
     
    fold = 0
    info = np.load(os.path.join(root_dir,"patient_info.pkl"), allow_pickle=True)
    ids = list(info.keys())

    train_keys, test_keys = get_split(fold, ids)
    train_data = load_dataset(train_keys, root_dir)
    
    data = load_dataset(ids, root_dir)
    nb_entities = len(data)
    types = [x for x in data[1].keys() if 'data' in x.split('_')]
    nb_samples = []
    print("Using data stored in ",root_dir)
    for typ in types:
        nb_samples.append([np.shape(data[x][typ])[0] for x in info.keys()])
#        nb_samples.append([np.shape(data[x][typ])[0] for x in range(1,nb_entities+1)])

    max_samples = max(nb_samples[0]+nb_samples[1])
    min_samples = min(nb_samples[0]+nb_samples[1])
    avg_samples = np.ceil(np.mean(nb_samples[0]+nb_samples[1]))
    all_samples_without_transformation = nb_entities * len(types) * avg_samples
    ALL_SAMPLES = all_samples_without_transformation
#    NUM_BATCHES = np.ceil(ALL_SAMPLES/BATCH_SIZE)
    NUM_BATCHES = int(cf['n_batches_per_epoch'])
    
    val_data = load_dataset(test_keys, root_dir)
    
    data_gen_validation = BatchGenerator(val_data, BATCH_SIZE, num_batches=NUM_BATCHES, seed=False,
                                            patch_size=INPUT_PATCH_SIZE)
    data_gen_validation = SingleThreadedAugmenter(data_gen_validation, Compose([ConvertSegToOnehotTransform(range(num_classes), seg_channel=0, output_key='seg_onehot')]))

    data_gen_training = create_data_gen(train_data, cf['BATCH_SIZE'], INPUT_PATCH_SIZE, NUM_BATCHES, cf["model_opts"][model_str]['n_classes'], phase="train", transforms=eval(cf["TRANSFORMS"]),do_elastic_transform=True, do_rotation=True, transform_prob=0.1)
    dataloaders = {'train': data_gen_training, 'val': data_gen_validation}
    model, train_history, val_history = train_model(model, optimizer_ft, exp_lr_scheduler, dataloaders, cf)
   
    with open("final_metrics.json") as f: 
        update_experiments_registry(str(experiment_key), json.loads(f.read()), register)

    now = datetime.now()
 
    header = 'loss history'
    np.savetxt(result_dir+"/val_history.csv", val_history, delimiter=",", fmt='%s', header=header)
    np.savetxt(result_dir+"/train_history.csv", train_history, delimiter=",", fmt='%s', header=header)
  
    torch.save(model, result_dir+'/entire_model_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.pth')
    
if __name__ == "__main__":
    setup_training('../examples/params.json', ids=6, root_dir='../examples/preprocessed/', result_dir='../examples/results_test', register="experiments.json", opts={})

