import torch
import os
import numpy as np
from mrinet.dataset.preprocessing import load_dataset, load_concepts_dataset
from mrinet.train_unet import create_data_gen, create_data_gen, get_split
from torch.autograd import Variable
import pickle as pkl
import json
from batchgenerators.augmentations.utils import resize_image_by_padding
import pandas as pd
#import matplotlib.pyplot as plt
#import cv2

# source: https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
def get_jacobian_old(net, x, noutputs):
#    x = x.squeeze()
#    n = x.size()[0]
#    x = x.repeat(noutputs, 1)
    x.requires_grad_(True)
    y = net(x)
    print(np.shape(y))
    a = torch.eye(noutputs)
    b = torch.eye(noutputs)
    c = torch.eye(noutputs)
    d = torch.eye(noutputs)

    grad_seq = torch.stack((a, b, c, d)).reshape(-1,4,164,164).double()
    y.backward(grad_seq)
    return x.grad.data

def get_jacobian(net, x, noutputs):
    x.requires_grad_(True)    
    out = net.fun(x)
    out.requires_grad_(True)
    out.retain_grad()

    y = net.forward_partial(out)

    a = torch.eye(noutputs)
    b = torch.eye(noutputs)
    c = torch.eye(noutputs)
    d = torch.eye(noutputs)

    grad_seq = torch.stack((a, b, c, d)).reshape(-1,4,164,164).double()
    

    y.backward(grad_seq)
    return out.grad.data


def activations(model, loader, layer_fun=None, n_sample=1, concepts=False):
    """
    Return Activations

    :param model: An nn.Module object whose activations we want to inspect.
    :param loader: A generator whose next() iterator returns (x, y).
    :param layer_fun: A function that returns a tensor of features, when
      applied to a model and input x.
    :param n_sample: The number of images from the loader to compute
      activations for.
    :returns summary: A list of dictionaries, each giving the image patch,
      feature activations for the first n_sample patches in
      the loader.

    Example
    -------
    >>> loader = patch_gen_train()
    >>> model = UNet(params["model_opts"])
    >>> model.load_state_dict(
          torch.load(model_path)
        )
    >>>
    >>> summary = activations(model, loader)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Getting activations...")
    model.double()
    if not layer_fun:
        layer_fun = lambda model, x: model.pre_pred(x)

    summary = []
    dat = next(loader)

    if not concepts:
        output_folder = './acts-grads-full-data'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not os.path.exists(output_folder + "/y"):
            os.mkdir(output_folder + "/y")
        if not os.path.exists(output_folder + "/x"):
            os.mkdir(output_folder + "/x")
        if not os.path.exists(output_folder + "/y_hat"):
            os.mkdir(output_folder + "/y_hat")
        if not os.path.exists(output_folder + "/grad"):
            os.mkdir(output_folder + "/grad")
        if not os.path.exists(output_folder + "/h"):
            os.mkdir(output_folder + "/h")


        PATCH_SIZE = (348,348)
        for i in range(0,len(dat['pathologies'])):
        
            #print(dat.keys())
            print("Patient: {}".format(dat['patient_ids'][i]))
            patient_ids = dat['patient_ids'][i]
            pathologies = dat['pathologies'][i]
            patient_id = dat['patient_ids'][i]
            frame_id = dat['frame_ids'][i]    
            
            for frame_suffix in ['ed','es']:
                #print(np.shape(dat[frame_suffix+'_data'][i]))
        #         plt.imshow(dat[frame_suffix+'_data'][i][0][0])
        #         plt.show()
                x = torch.from_numpy(dat[frame_suffix+'_data'][i].astype(np.double)).to(device)
                seg_onehot = dat[frame_suffix+'_seg'][i]
                y = torch.from_numpy(dat[frame_suffix+'_seg'][i]).to(device) # we need an output encoded into 4 layers          
                y_pred1 = model.forward(x)
                jacobian = get_jacobian(model, x, 164) # 164 number of output pixels
                fileObject = open("{}/grad/{}-{}-{}-grad.pkl".format(output_folder,patient_id,frame_id,frame_suffix.upper()), 'wb')
                pkl.dump(jacobian, fileObject)
                fileObject.close()

                y_hat1 = np.argmax(y_pred1.cpu().detach().numpy(), axis=1)
        
                shp_ed = np.shape(dat[frame_suffix+'_seg'])
                xd = resize_image_by_padding(y_hat1[0], (max(shp_ed[1], PATCH_SIZE[0]), max(shp_ed[2], PATCH_SIZE[1])), pad_value=0)
        
                xuu = xd.astype('float32') # convert so the types match
                xu = x.cpu().detach().numpy().reshape(348,348)
#                save_nii(output_folder + '/x/', str(patient_id) + '_' + str(frame_id), frame_suffix.upper(),x)    
                fileObject = open("{}/x/{}-{}-{}-X.pkl".format(output_folder,patient_id,frame_id,frame_suffix.upper()), 'wb')
                pkl.dump(x, fileObject)
                fileObject.close()

#                save_nii(output_folder + '/pred/', str(patient_id) + '_' + str(frame_id), frame_suffix.upper(), xuu)    
                fileObject = open("{}/y_hat/{}-{}-{}-Y_hat.pkl".format(output_folder,patient_id, frame_id, frame_suffix.upper()), 'wb')
                pkl.dump(xuu, fileObject)
                fileObject.close()

#                save_nii(output_folder + '/gt/', str(patient_id) + '_' + str(frame_id), frame_suffix.upper(),  y.cpu().detach().numpy().reshape(348,348))  
                fileObject = open("{}/y/{}-{}-{}-Y.pkl".format(output_folder,patient_id, frame_id, frame_suffix.upper()), 'wb')
                pkl.dump(y.cpu().detach().numpy().reshape(348,348), fileObject)
                fileObject.close()

                h = layer_fun(model, x)
                fileObject = open("{}/h/{}-{}-{}-H.pkl".format(output_folder,patient_id, frame_id, frame_suffix.upper()), 'wb')
                pkl.dump(h.cpu().detach().numpy(), fileObject)
                fileObject.close()

#                np.savetxt(output_folder + "/h/{}-{}-{}-H.csv".format(patient_id, frame_id, frame_suffix.upper()), h.detach().numpy(), delimiter=",")
        #    break
        df = pd.DataFrame()
        df["patient_ids"] = dat["patient_ids"] 
        df["pathologies"] = dat["pathologies"] 
        df["frame_ids"] = dat["frame_ids"]
        df.to_csv (output_folder + '/patients_info.csv', index=False, header=True)

    else:

        for i, D in enumerate(loader):
    
    
            print("batch {}".format(i))
            if i > n_sample:
                break
            print(D.keys())
            if not concepts:
                # each element of the batch
                x = torch.from_numpy(np.array(D['ed_data'])).to(device)
                #types = D['types']
                patient_ids = D['patient_ids']
                pathologies = D['pathologies']
                #seg_onehot = D['seg_onehot']
                y = torch.from_numpy(np.array(D['ed_seg'])).to(device) # we need an output encoded into 4 layers          
            else:       
                # each element of the batch
                sample = np.random.choice(range(len(D['patient_ids'])), 10)
               
                x = torch.from_numpy(np.array(D['segments'])[sample].reshape(-1,1,348,348).astype(np.double)).to(device)
                types = np.array(D['types'])[sample]
                patient_ids = np.array(D['patient_ids'])[sample]
                pathologies = np.array(D['pathologies'])[sample]
    
            y_pred1 = model.forward(x)
            #print("Predictions shape: ",np.shape(y_pred1))
            y_hat1 = np.argmax(y_pred1.cpu().detach().numpy(), axis=1)
     
            h = layer_fun(model, x)
            for j, _ in enumerate(x):
                if not concepts:
                    summary.append({
                        "x": x[j].detach().cpu().numpy(),
                        "h": h[j].detach().cpu().numpy(),
                        "y": D['seg'][j],
                        "y_hat": y_hat1[j],
                    })
                else:    
                    summary.append({
                        "x": x[j].detach().cpu().numpy(),
                        "h": h[j].detach().cpu().numpy(),
                        "y_hat": y_hat1[j],
                        "types": types,
                        "pathologies": pathologies,
                        "patient_ids": patient_ids
                })

    return summary


def ft_fun(model, x):
    """
    Activations to write for the U-net
    """
    x, conv1_out, conv1_dim = model.down_1(x)
    x, conv2_out, conv2_dim = model.down_2(x)
    x, conv3_out, conv3_dim = model.down_3(x)
    x, conv4_out, conv4_dim = model.down_4(x)

    # norms for each feature map at the encoder layer
    x = model.conv5_block(x)
    return x
    #return torch.sum(x ** 2, dim=(2, 3)) ** (0.5)

def reshape_activations(summary, concepts=False):
    """
    Matrix-ify the output of activations()

    :param summary: The output of the activations function. This is a list of
      dictionaries describing each patch.
    :return: A tuple with the following components,
      H: A numpy array whose rows correspond to patches and whose values are
      activations.

    Example
    -------
    >>> summary = activations(model, loader)
    >>> H = reshape_activations(summary)
    >>>
    >>> np.savetxt(os.path.join(params["save_dir"], "H.csv"), H, delim=",")
    """
    print("Entering reshape activations")
    N = len(summary)
    K = len(summary[0]["h"])
    P = np.shape(summary[0]["x"])
    if not concepts:
        Q = np.shape(summary[0]["y"])
    R = np.shape(summary[0]["y_hat"])

    #print(P,Q,R)
    #print(np.shape(summary[0]["x"]), np.shape(summary[0]["y"]), np.shape(summary[0]["y_hat"]),)
   # print(np.shape(summary[0]["y_hat"]))
    H = np.zeros((N, K))
    X = np.zeros((N, P[0], P[1], P[2]))
    Y_hat = np.zeros((N, R[0],R[1]))
    if not concepts:
        Y = np.zeros((N, Q[0],Q[1],Q[2]))
    #print(np.shape(R))

    for i in range(N):
        H[i] = summary[i]["h"]
        X[i] = summary[i]["x"]
        if not concepts:
            Y[i] = summary[i]["y"]
        Y_hat[i] = summary[i]["y_hat"]

    if not concepts:
        return H, X, Y, Y_hat
    return H, X, Y_hat



def save_summary(model, loader, layer_fun, out_path, concepts=False, **kwargs):
    print("Saving summary...")
    i = 0
    while i != 2000:
        summary = activations(model, loader, layer_fun, n_sample=1, concepts=concepts)
        if not concepts:
            H, X, Y, Y_hat = reshape_activations(summary)
        else:
            H, X, Y_hat = reshape_activations(summary, concepts=concepts)

        np.savetxt("{}-{}-H.csv".format(out_path,i), H, delimiter=",")
    
        fileObject = open("{}-{}-X.pkl".format(out_path,i), 'wb')
        pkl.dump(X, fileObject)
        fileObject.close()

        if not concepts:    
            fileObject = open("{}-{}-Y.pkl".format(out_path,i), 'wb')
            pkl.dump(Y, fileObject)
            fileObject.close()
    
        fileObject = open("{}-{}-Y_hat.pkl".format(out_path,i), 'wb')
        pkl.dump(Y_hat, fileObject)
        fileObject.close()
        i+=1


def collect_activations_from_train_data(path, model_path, model_str="duke_unet"):
    with open(path) as f:
        tmp = f.read()
        cf = json.loads(tmp)

#    cf = {
#        'BATCH_SIZE': 1,
#        'INPUT_PATCH_SIZE': (348, 348), # calculated using notebook: Calculate-U-Net Size
#        'num_classes': 4,
#        'EXPERIMENT_NAME': 'Test-Implementation',
#        'results_dir': './results-experiment',
#        'n_epochs': 5,
#        'lr_decay': np.float32(0.985),
#        'base_lr': np.float32(0.0005),
#        'n_batches_per_epoch': 10,
#        'n_test_batches': 10,
#        'n_feedbacks_per_epoch': 10,
#        'num_workers': 6,
#        'workers_seeds': [123, 1234, 12345, 123456, 1234567, 12345678],
#        'weight_decay': 1e-5,
#        'n_input_channels': 1
#    }
    cf["BATCH_SIZE"] = 1 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold = 0
#    train_keys, test_keys = get_split(fold)
#    train_data = load_dataset(train_keys)
    
    train_data = load_dataset()
    nb_entities = len(train_data)
    types = [x for x in train_data[1].keys() if 'data' in x.split('_')]
    nb_samples = []
    for typ in types:
        nb_samples.append([np.shape(train_data[x][typ])[0] for x in range(1,nb_entities+1)])
    max_samples = max(nb_samples[0]+nb_samples[1])
    min_samples = min(nb_samples[0]+nb_samples[1])
    avg_samples = np.ceil(np.mean(nb_samples[0]+nb_samples[1]))
    all_samples_without_transformation = nb_entities * len(types) * avg_samples
    ALL_SAMPLES = all_samples_without_transformation
    NUM_BATCHES = np.ceil(ALL_SAMPLES/cf["BATCH_SIZE"])
    NUM_BATCHES = 100
    
    model = torch.load(model_path, map_location=device)
    data_gen_training = create_data_gen(train_data, cf['BATCH_SIZE'], (cf['INPUT_PATCH_SIZE'],cf['INPUT_PATCH_SIZE']), NUM_BATCHES, cf["model_opts"][model_str+"_opts"]['n_classes'],  "eval")
    
    s = activations(model, data_gen_training, ft_fun, n_sample=1, concepts=False)


    # save activations to file
#    save_summary(
#        model,
#        data_gen_training,
#        ft_fun,
#        "./activations/activations",
#        n_sample=1
#    )

def collect_activations_from_concepts_data():
    cf = {
        'BATCH_SIZE': 1,
        'INPUT_PATCH_SIZE': (348, 348), # calculated using notebook: Calculate-U-Net Size
        'num_classes': 4,
        'EXPERIMENT_NAME': 'Test-Implementation',
        'results_dir': './results-experiment',
        'n_epochs': 5,
        'lr_decay': np.float32(0.985),
        'base_lr': np.float32(0.0005),
        'n_batches_per_epoch': 10,
        'n_test_batches': 10,
        'n_feedbacks_per_epoch': 10,
        'num_workers': 6,
        'workers_seeds': [123, 1234, 12345, 123456, 1234567, 12345678],
        'weight_decay': 1e-5,
        'n_input_channels': 1
    }
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold = 0
    train_data = load_concepts_dataset()
    
#    nb_entities = len(train_data)
#    types = [x for x in train_data[1].keys() if 'data' in x.split('_')]
#    nb_samples = []
#    for typ in types:
#        nb_samples.append([np.shape(train_data[x][typ])[0] for x in range(1,nb_entities+1)])
#    max_samples = max(nb_samples[0]+nb_samples[1])
#    min_samples = min(nb_samples[0]+nb_samples[1])
#    avg_samples = np.ceil(np.mean(nb_samples[0]+nb_samples[1]))
#    all_samples_without_transformation = nb_entities * len(types) * avg_samples
#    ALL_SAMPLES = all_samples_without_transformation
#    NUM_BATCHES = np.ceil(ALL_SAMPLES/cf["BATCH_SIZE"])
    NUM_BATCHES = 100
    
    
    model = torch.load('./results-7th-unet-step/intermediate_model_epoch_295.pth', map_location=device)
    data_gen_training = create_data_gen(train_data, cf['BATCH_SIZE'], cf['INPUT_PATCH_SIZE'], NUM_BATCHES, cf['num_classes'], "concepts")
    
    
    
    # save activations to file
    save_summary(
        model,
        data_gen_training,
        ft_fun,
        "./activations/activations",
        n_sample=20,
        concepts=True
    )

def check_sizes_helper():
    train_data = load_concepts_dataset()
    for i, x in train_data.items():
        print(i)
        print(len(x['pathology']))
        print(np.shape(x['segments']))


if __name__ == "__main__":
    path = "/media/adri/Adri-687Gb/Studies/PhD_UCD/notebooks/params14th_model.json"#"/home/people/19203757/unet-training/params.json"
    model_path = "/media/adri/Adri-687Gb/Studies/PhD_UCD/notebooks/model_14th_intermediate_model_epoch_125.pth" #'/home/people/19203757/unet-training/unet-11th-results/intermediate_model_epoch_175.pth'
    collect_activations_from_train_data(path, model_path)
    #collect_activations_from_concepts_data()





