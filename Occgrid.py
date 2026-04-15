import torch
import os
#import check_status


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
    print('---> loading models')
    ckpt = torch.load(src+mdl,map_location=torch.device('cpu'))
    for key1 in ckpt.keys():
        #print(key1)
        for key in ckpt[key1].keys():
            print(key, ckpt[key1][key].size())
    #print('--->',mdl,'/',ckpt['model']["ray_sampler.occs"].size())
    #print(ckpt['model']['ray_sampler.occs'])
    print('--->end loading')
    return ckpt


def remove_component(ckpt_path, component_to_remove,mdl):
#def remove_component(checkpoint, component_to_remove):
    # Load the checkpoint

    checkpoint = _load_ckpt(ckpt_path,'/model.ckpt')
    # Remove the component
    for component in component_to_remove:
        if component in checkpoint['model']:
            print('Yes, it has',component)
            del checkpoint['model'][component]
    
    # Save the modified checkpoint
    torch.save(checkpoint, ckpt_path+ mdl)
    _load_ckpt(ckpt_path,mdl)
    _get_size(source,mdl)

def read_occ(root):
    occ = []
    #for folder in os.listdir(root):
    ph = root+'nerf_synthetic/'
    print('ph:', ph)
    for sub in os.listdir(ph):
        pth = ph +sub+"/Tri-MipRF/"
        print(pth)
        for item in os.listdir(pth):
            if item[:4] == '2025':
                print(pth + item)
                ckpt = _load_ckpt(pth+item+'/','model.ckpt')
                occ.append(ckpt['model']['ray_sampler.occs'])
    print("lenght of lcc:", len(occ))
    return occ

def max_occ(root):
    occ = torch.stack(read_occ(root))
    print('occ size:',occ.size())

    # Use torch.max to get the maximum value at each position
    max_tensor, _ = torch.max(occ, dim=0)
    print("max tensor: ",max_tensor)
    print("Max_tensor size check: ",max_tensor.size())
    return max_tensor

def replace_occ(ckpt_pth,maxOcc):
    ckpt = _load_ckpt(ckpt_pth,'/model.ckpt')
    ckpt['model']['ray_sampler.occs'] = maxOcc
    print(ckpt['model']['ray_sampler.occs'])
    torch.save(ckpt,ckpt_pth+'/occ_test.ckpt')    

'''
#source_f60 = "/home/jzhou23/Tri-MipRF-CP/log_cp/nerf_synthetic/lego_f60/Tri-MipRF/2024-07-18_19-28-36"
source = './log_sample/fd3/test/'
mdl = '/noob.ckpt'
#check size
#root = "./log_worker/fd1/nerf_synthetic/"
component_to_remove = ['field.encoding.x','field.encoding.y','field.encoding.z','ray_sampler.occs', 'ray_sampler._binary','field.mlp_base.params', 'field.mlp_head.params']
#component_to_remove = ['field.mlp_base.params', 'field.mlp_head.params'] #,'ray_sampler._binary']
#_get_size(source,mdl)
#ckpt = _load_ckpt(source,'model.ckpt')
for i in ['field.mlp_base.params', 'field.mlp_head.params']:
    print(ckpt['model'][i])
#_load_ckpt(source_f60,mdl)
#remove_component(source, component_to_remove,mdl)
'''
#read_occ(root)
#max_occ(root)

'''
#maxocc = max_occ(root)
#maxocc = torch.ones(torch.Size([2097152]))
for i in range(2,5):
        #pth = root+folder+"/Tri-MipRF/"
        pth = root+"worker_f"+str(i)+"/Tri-MipRF/"
        for item in os.listdir(pth):
            print(pth+item+'/')
            replace_occ(pth+item+'/',maxocc)
'''