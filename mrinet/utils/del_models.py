import os

def delete_temporary_models(root):
    while True:
        x = os.listdir(root)
        filtered = [z for z in x if 'intermediate_model_epoch_' in z]
        
        if len(filtered) > 2:
            nums = [int(x.split('.')[0].split('_')[-1]) for x in filtered]
            maxi = max(nums)
            for f in filtered:
                if int(f.split('.')[0].split('_')[-1]) < maxi:
                    os.remove(root+f)


if __name__ == "__main__":
    root = '/home/people/19203757/unet-training/'
    delete_temporary_models(root)
    
                   
