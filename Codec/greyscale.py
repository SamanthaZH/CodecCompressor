import numpy as np
#from sklearn.preprocessing import normalize
import os
import torch
from PIL import Image 
import collections
import pickle

def grayscale(ph,name):
  nm = name
  src = ph + "para_folder/"
  dst = ph +"grey/"
  #dst = ph+"jpeg60/"
   
  os.makedirs(ph+"/max_min/",exist_ok=True)

  density_arr = ['density_plane-','density_line-']
  app_arr = ['app_plane-','app_line-']

  # density plane
  mini = collections.OrderedDict()
  maxm = collections.OrderedDict()

  for arr_name in density_arr:
    for i in range(0,3):
      data = np.load(src+arr_name+str(i)+".npy")
      
      ra = (data.shape)[1]
      print(ra)
      os.makedirs(dst+arr_name+str(i),exist_ok= True)
    
      pth   = dst+arr_name+str(i)
      #print("load succeed!")
      print("-------------------------------")
      print(arr_name+str(i))
      #print(type(data.shape)) 
      #print(data)
      #normalize data
      data_min = data.min(axis=(2,3), keepdims=True)
      data_max = data.max(axis=(2,3), keepdims=True)
      data_norm = (data - data_min)/(data_max - data_min)
      mini[arr_name+str(i)] = data_min
      maxm[arr_name+str(i)] = data_max

      #print(np.round((data_norm *255)))
      #print(data_norm*255)
      ## read as grey scale
    
      #data_grey = np.round((data_norm *255).squeeze()).astype(int)
      data_grey = np.round((data_norm *255).squeeze()).astype(np.uint8)
      #print(data_grey.shape)

      #save to new file as inference 
      #print(ra)
      for channel_idx in range(ra):    #density 16
        planes = data_grey[channel_idx]
        #print(type(planes))
        #print(planes.shape)
        plane = np.asarray(planes, dtype=np.uint8)

        #print(plane.size)
        plane_img = Image.fromarray(plane)
        plane_img=plane_img.convert('L')
        #print(plane_img)
        plane_img.save(pth+'/'+arr_name+str(channel_idx)+".png")
        #plane_img.save(pth+'/'+arr_name+str(channel_idx)+".jpg","JPEG",quality=60)
        #plane_img.show()

  with open(ph+"max_min/density_min.pkl","wb") as f:
    pickle.dump(mini, f)
  with open(ph+"max_min/density_max.pkl","wb") as f:
    pickle.dump(maxm, f)


  # Appearance plane
  mini2 = collections.OrderedDict()
  maxm2 = collections.OrderedDict()

  for arr_name in app_arr:
    for i in range(0,3):
      data = np.load(src+arr_name+str(i)+".npy")
      ra = (data.shape)[1]
      os.makedirs(dst+arr_name+str(i),exist_ok= True)
    
      pth   = dst+arr_name+str(i)
      #print("load succeed!")
      print("-------------------------------")
      print(arr_name+str(i))
      #print(type(data.shape)) 
    
      #normalize data
      data_min = data.min(axis=(2,3), keepdims=True)
      data_max = data.max(axis=(2,3), keepdims=True)
      data_norm = (data - data_min)/(data_max - data_min)
      mini2[arr_name+str(i)] = data_min
      maxm2[arr_name+str(i)] = data_max

      ## read as grey scale
      #print(data_norm.shape)
      #print(((data_norm*255).squeeze()).shape)
      data_grey = np.round((data_norm *255).squeeze()).astype(np.uint8)
      #print(data_grey.shape)
    
      #save to new file as inference 
      print(ra)
      for channel_idx in range(ra):    #appearance 48/24/12
        planes = data_grey[channel_idx]
        #print(type(planes))
        #print(planes.shape)
        plane = np.asarray(planes, dtype=np.uint8)
        #print(plane.size)
        plane_img = Image.fromarray(plane)
      
        plane_img=plane_img.convert('L')
      
        plane_img.save(pth+'/'+arr_name+str(channel_idx)+".png")
        #plane_img.save(pth+'/'+arr_name+str(channel_idx)+".jpg","JPEG",quality =60)
        #plane_img.show()

  with open(ph+"max_min/app_min.pkl","wb") as f:
    pickle.dump(mini2, f)
  with open(ph+"max_min/app_max.pkl","wb") as f:
    pickle.dump(maxm2, f)

def batch_gray(root_dir,name):
  for i in range(1,15):
    nm = "dynamic_"+name+"_" + str(i)+"f"+str(i+1)
    pth = root_dir+nm+"/"
    
    print("======="+nm+"=========")
    grayscale(pth,nm)

nm = "lego"
name = "dynamic_"+nm +"_2f3"
root_dir = "./log/"#+name+"/"

#ph ="/content/drive/MyDrive/NeRF/TensoRF_main/log/bubasaur/dynamic/dynamic_bulbasaur_20f21/"
ph = root_dir+name+"/"
#grayscale(ph,name)
#batch_gray(root_dir,nm)
#pad(ph)

