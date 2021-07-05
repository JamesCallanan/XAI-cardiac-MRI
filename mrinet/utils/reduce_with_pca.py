import csv
import numpy as np
import os
from sklearn.decomposition import IncrementalPCA
import pickle as pkl


def reduce_pca():
    #"activations/activations-1-H.csv"
    
    root = './activations/'
    ipca = IncrementalPCA(n_components=2, batch_size=3)
    
    fls = [x for x in os.listdir(root) if x.endswith('.csv')]
    for i,fname in enumerate(fls):
        with open(root+fname) as csv:
            x = csv.read()
            
        dd = [float(v)  for c in x.split(',') for v in  c.split('\n') if v]        
        activations = np.array(dd).reshape((int(np.shape(dd)[0]/512),512))
        ipca.partial_fit(activations)
    
    # save the model to disk
    filename = 'ipca_model.sav'
    pkl.dump(ipca, open(filename, 'wb'))
        
    for i,fname in enumerate(fls):
        with open(root+fname) as csv:
            x = csv.read()
            
        dd = [float(v)  for c in x.split(',') for v in  c.split('\n') if v]        
        activations = np.array(dd).reshape((int(np.shape(dd)[0]/512),512))
        transformed = ipca.transform(activations) # saved transformed activations
        outfile = './transformed/'+fname.split('.')[0] + "_transformed"
        np.save(outfile, transformed)
   
if __name__ == "__main__":
    reduce_pca()
