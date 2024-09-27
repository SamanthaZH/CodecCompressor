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
    #print('--->',mdl)
    #ckpt_f1 = torch.load(src+"fm.ckpt")
    #print(ckpt_f1.size())
    ckpt = torch.load(src+mdl,map_location=torch.device('cpu'))
    #print(ckpt_f1)
    print(ckpt.keys())
    for key1 in ckpt.keys():
        print(key1)
        for key in ckpt[key1].keys():
            print(key1, ckpt[key1][key].size())
    #print('--->',mdl,'/',ckpt['model']["ray_sampler.occs"].size())
    #print(ckpt['model']['ray_sampler.occs'])
    return ckpt


#def remove_component(ckpt_path, component_to_remove):
def remove_component(checkpoint, component_to_remove):
    # Load the checkpoint

    #checkpoint = torch.load(ckpt_path+'/model.ckpt')
    #checkpoint = _load_ckpt(ckpt_path,'/model.ckpt')
    # Remove the component
    for component in component_to_remove:
        if component in checkpoint['model']:
            print('Yes, it has',component)
            del checkpoint['model'][component]
    
    # Save the modified checkpoint
    torch.save(checkpoint, ckpt_path+'/new_model.ckpt')

def read_occ(root):
    occ = []
    for folder in os.listdir(root):
        pth = root+folder+"/Tri-MipRF/"
        for item in os.listdir(pth):
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
    torch.save(ckpt,ckpt_pth+'/occ.ckpt')    

#source_f60 = "/home/jzhou23/Tri-MipRF-CP/log_cp/nerf_synthetic/lego_f60/Tri-MipRF/2024-07-18_19-28-36"
#source = '/home/jzhou23/Tri-MipRF-CP/log_cp/nerf_synthetic/lego_f10/Tri-MipRF/2024-06-21_15-46-48'
#mdl = '/model.ckpt'
#root = "./log_worker/fd1/nerf_synthetic/"
#component_to_remove = ['field.encoding.x','field.encoding.y','field.encoding.z']
#_get_size(source,mdl)
#_load_ckpt(source,mdl)
#_load_ckpt(source_f60,mdl)
#remove_component(source, component_to_remove)
#read_occ(root)
#max_occ(root)

'''
maxocc = max_occ(root)
#for folder in os.listdir(root):
for i in range(2,61):
        #pth = root+folder+"/Tri-MipRF/"
        pth = root+"worker_f"+str(i)+"/Tri-MipRF/"
        for item in os.listdir(pth):
            print(pth+item+'/')
            replace_occ(pth+item+'/',maxocc)
'''