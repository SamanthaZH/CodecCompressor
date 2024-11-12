import os, collections
import torch
import numpy as np
import subprocess
import Codec.mergePlane as mp



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loading(root_dir,name):        
  print("------- "+name+" output----------")
  nmm = name
  #pth = root_dir+nm+"/"
  pth = root_dir
  
  #print(pth)
  print(nmm+".th")
  ckpt = torch.load(pth+nmm+".th", map_location=device)
  os.makedirs(pth+nmm+"/para_folder/", exist_ok = True)
  #p = "/content/drive/MyDrive/NeRF/TensoRF_main/figs/para_folder/"
        
  for name in ckpt['state_dict']:
    print(str(name))
    #print(ckpt['state_dict'][name].size)
    st = str(name)
    r = "-"
    nm = st.replace(".",r)
    #print(nm)
    print(ckpt["state_dict"][name].cpu().numpy().shape)
    #with open("/content/drive/MyDrive/NeRF/TensoRF_main/log/dynamic/result_grey/paras_folder/"+nm+".npy",'wb') as ff:
    with open(pth+nmm+"/para_folder/"+nm+".npy",'wb') as ff:
      np.save(ff, ckpt["state_dict"][name].cpu().numpy())
  print("Saved-npy parameters ")

def batch_loading(root,scene):
    saved_files = collections.defaultdict(list)
    root = 'log/fd1_worker/worker_f'
    for idx in range(2,scene+1):
        root_pth = root + str(idx) +'/Tri-MipRF/'
        for file in os.listdir(root_pth):
          ckpt = mp.load_model(root_pth+file+'/')
          print('-------------', f"worker_f{idx}",'-------------')
          #print(ckpt['model'].keys())
          
        os.makedirs(root_pth+"para_folder/", exist_ok = True)
        
        plane_list = ['field.encoding.x','field.encoding.y','field.encoding.z']
        for name_sub in plane_list:
                # Replace '.' in parameter names with '-'
                nm2 = str(name_sub).replace(".", "-")
                param_path = root_pth + "para_folder/" + nm2 + ".npy"
                
                print(ckpt["model"][name_sub].cpu().numpy().shape)
                # Save the parameter as a .npy file in the created directory
                #with open(param_path, 'wb') as ff:
                  #  np.save(ff, ckpt["model"][name_sub].cpu().numpy())
                
                # Append the saved file path to the list
                saved_files.append(param_path)

    print("Saved-npy parameters ")

# Running
#batch_loading(root_dir,name)
#loading(root_dir,name)