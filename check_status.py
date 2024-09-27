import torch
import os
from Occgrid import remove_component

def _get_size(src,mdl):
        print("---> size: ", mdl)

        # Path to your checkpoint file
        ckpt_file_path = src+mdl

        # Get the file size in bytes
        file_size = os.path.getsize(ckpt_file_path)

        # Convert the file size to megabytes (optional)
        file_size_mb = file_size / (1024 * 1024)

        print(f"Checkpoint file size: {file_size} bytes")
        print(f"Checkpoint file size: {file_size_mb:.2f} MB")

def _load_ckpt(src,mdl):
    #load from path src and mdl(model.ckpt or occ.ckpt)
    #print('--->',mdl)
    #ckpt_f1 = torch.load(src+"fm.ckpt")
    #print(ckpt_f1.size())
    ckpt = torch.load(src+mdl,map_location=torch.device('cpu'))
    #print(ckpt_f1)
    print(ckpt.keys())
    
    if mdl == "new_model.ckpt":
        for key1 in ckpt.keys():
            print(key1)
            for key in ckpt[key1].keys():
                print(key, ckpt[key1][key].size())
        #print('--->',mdl,'/',ckpt['model']["ray_sampler.occs"].size())
    
        #print(ckpt['model']['ray_sampler.occs'])
    return ckpt

def remove_component(ckpt_path, component_to_remove):
    # Load the checkpoint

    #checkpoint = torch.load(ckpt_path+'/model.ckpt')
    checkpoint = _load_ckpt(ckpt_path,'/model.pth')
    # Remove the component
    for component in component_to_remove:
        if component in checkpoint:
            print('Yes, it has',component)
            #del checkpoint['model'][component]
            del checkpoint[component]
    
    # Save the modified checkpoint
    torch.save(checkpoint, ckpt_path+'/new_model.ckpt')

#kplanes-model
ckpt_pth = 'log/kplanes/coffee/hybrid/'#hybrid/'
_load_ckpt(ckpt_pth,'new_model.ckpt')
#remove_component_list = ['optimizer', 'lr_scheduler', 'global_step']
#remove_component(ckpt_pth, remove_component_list)
