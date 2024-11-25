#import cv2
from PIL import Image as img
import numpy as np
import os
import imageio
import io
import yuvio
import glob
import collections,pickle
import torch

def yuv(src,nm):
  #this function is to convert greyscale to yuv444 format
  print("*****call YUV********")
  pth = src + "/grey/"
  dst = src+"/yuv/"
  os.makedirs(dst,exist_ok=True)

  arr = ['density_plane-','app_plane-']
  arr_size=[]
  for i in range(len(arr)):
    yuv_pth = dst+arr[i].split('-')[0]+'/'
    #con = "concat:"
    os.makedirs(yuv_pth,exist_ok=True)
    print(yuv_pth)
    Y = arr[i]+str(0)
    U = arr[i]+str(1)
    V = arr[i]+str(2)
    print(Y,U,V)
    
    
    y = np.zeros((378,378),dtype = np.uint8)
    u = np.zeros((378,378),dtype = np.uint8)
    v = np.zeros((378,378),dtype = np.uint8)
    
    #pig
    #y = np.zeros((400,400),dtype = np.uint8)
    #u = np.zeros((400,400),dtype = np.uint8)
    #v = np.zeros((400,400),dtype = np.uint8)
    
    if nm == 'lego':   #lego
      y = np.zeros((420,420),dtype = np.uint8)
      u = np.zeros((420,420),dtype = np.uint8)
      v = np.zeros((420,420),dtype = np.uint8)
    if nm == 'chair':
      y = np.zeros((388,388),dtype = np.uint8)
      u = np.zeros((388,388),dtype = np.uint8)
      v = np.zeros((388,388),dtype = np.uint8)
    if nm == 'hotdog':   #hotdog
      y = np.zeros((467,467),dtype = np.uint8)
      u = np.zeros((467,467),dtype = np.uint8)
      v = np.zeros((467,467),dtype = np.uint8)   
    if nm == 'drum': #drum
      y = np.zeros((378,378),dtype = np.uint8)
      u = np.zeros((378,378),dtype = np.uint8)
      v = np.zeros((378,378),dtype = np.uint8)
    if nm =='mic': #mic
      y = np.zeros((306,306),dtype = np.uint8)
      u = np.zeros((306,306),dtype = np.uint8)
      v = np.zeros((306,306),dtype = np.uint8)
    if nm == 'ship': #ship
      y = np.zeros((368,368),dtype = np.uint8)
      u = np.zeros((368,368),dtype = np.uint8)
      v = np.zeros((368,368),dtype = np.uint8)
    if nm == 'materials': #ship
      y = np.zeros((531,531),dtype = np.uint8)
      u = np.zeros((531,531),dtype = np.uint8)
      v = np.zeros((531,531),dtype = np.uint8)
    if nm == 'ficus': #ship
      y = np.zeros((442,442),dtype = np.uint8)
      u = np.zeros((442,442),dtype = np.uint8)
      v = np.zeros((442,442),dtype = np.uint8)

    #frame = yuvio.empty(420, 420, "yuv444p")
    for files in os.listdir(pth+Y): 
      fn = files.split('.')[0]
      #print(fn)
      #print(pth+Y+'/'+files)
      #print(pth+U+'/'+files)
      #print(pth+V+'/'+files)
      gray_y = np.array(img.open(pth+Y+'/'+files)) 
      gray_u = np.array(img.open(pth+U+'/'+files))
      gray_v = np.array(img.open(pth+V+'/'+files))
      
      '''
      arr_size.append(gray_y.shape)
      arr_size.append(gray_u.shape)
      arr_size.append(gray_v.shape)

      y[:gray_y.shape[0],:gray_y.shape[1]] = gray_y
      y[:gray_u.shape[0],:gray_u.shape[1]] = gray_u
      y[:gray_v.shape[0],:gray_v.shape[1]] = gray_v
      '''
      if nm == 'hotdog' :
        y[:467, :463] = gray_y
        u[:124, :463] = gray_u
        v[:124, :467] = gray_v
      if nm == 'drum':
        y[:289, :378] = gray_y
        u[:246, :378] = gray_u
        v[:246, :289] = gray_v
      if nm == 'lego':
        y[:420, :237] = gray_y
        u[:270, :237] = gray_u
        v[:270, :420] = gray_v
      if nm == 'mic':
        y[:303, :306] = gray_y
        u[:289, :306] = gray_u
        v[:289, :303] = gray_v
      if nm == 'ship':
        y[:365, :368] = gray_y
        u[:200, :368] = gray_u
        v[:200, :365] = gray_v
      if nm == 'materials':
        y[:421, :531] = gray_y
        u[:120, :531] = gray_u
        v[:120, :421] = gray_v
      if nm == 'ficus':
        y[:294, :207] = gray_y
        u[:442, :207] = gray_u
        v[:442, :294] = gray_v
      if nm == 'chair':
        y[:270, :257] = gray_y
        u[:388, :257] = gray_u
        v[:388, :270] = gray_v

      
      #y[:378, :284] = gray_y
      #u[:250, :284] = gray_u
     # v[:250, :378] = gray_v
      
      frame_444 = yuvio.frame((y, u, v), "yuv444p")
      print(frame_444.y.shape)
      #print(yuv_pth+files.split('.')[0]+'.yuv')
      
      yuvio.imwrite(yuv_pth+fn+'.yuv', frame_444)
  return arr_size[:3]

def yuv_llff(src):
  #this function is to convert greyscale to yuv444 format
  pth = src + "/grey/"
  dst = src+"/yuv/"
  os.makedirs(dst,exist_ok=True)

  arr = ['density_plane-','app_plane-']
  
  for i in range(len(arr)):
    yuv_pth = dst+arr[i].split('-')[0]+'/'
    #con = "concat:"
    os.makedirs(yuv_pth,exist_ok=True)
    print(yuv_pth)
    Y = arr[i]+str(0)
    U = arr[i]+str(1)
    V = arr[i]+str(2)
    print(Y,U,V)

    y = np.zeros((786,786),dtype = np.uint8)
    u = np.zeros((786,786),dtype = np.uint8)
    v = np.zeros((786,786),dtype = np.uint8) 

    '''
    if nm == 'flower' or 'horns' or 'trex':
        y[:786, :706] = gray_y
        u[:471, :706] = gray_u
        v[:471, :786] = gray_v
    '''
    #frame = yuvio.empty(420, 420, "yuv444p")
    for files in os.listdir(pth+Y): 
      fn = files.split('.')[0]
      #print(fn)
      pthy = pth+Y+'/'
      pthu = pth+U+'/'
      pthv = pth+V+'/'
      if os.path.exists(pthy+files):
        gray_y = np.array(img.open(pthy+files)) 
        y[:786,:706] = gray_y
      else:
        y =y
      
      
      if os.path.exists(pthu+files):
        gray_u = np.array(img.open(pthu+files))
        u[:471,:706] = gray_u
      else:
        u=u
      
      if os.path.exists(pthu+files):
        gray_v = np.array(img.open(pthv+files))
        v[:471,:786] = gray_v
      else:
        v=v
      #print(gray_u)
      #gray_y = cv2.imread(pth+Y+'/'+files, cv2.IMREAD_GRAYSCALE)
      #gray_u = cv2.imread(pth+U+'/'+files, cv2.IMREAD_GRAYSCALE)
      #print(gray_u.shape)
      #gray_v = cv2.imread(pth+V+'/'+files, cv2.IMREAD_GRAYSCALE)
      #print(gray_v.shape)     
      
      
      frame_444 = yuvio.frame((y, u, v), "yuv444p")
      #print(frame_444.y)
      #print(yuv_pth+files.split('.')[0]+'.yuv')
      
      yuvio.imwrite(yuv_pth+fn+'.yuv', frame_444)

def batch_yuv(root_dir,nm):
  arr = {}
  for i in range(0,2):
    name = "dynamic_"+nm+"_"+str(i)+"f"+str(i+1)
    pth = root_dir+name
    print("=============="+name+"==================")
    #size = yuv(pth,name)
    #arr[name]=size
    print(arr[name][0])
    print('--Yeah! Saved frames--')
  #with open(root_dir+'data.pkl', 'wb') as f:
  #  pickle.dump(arr, f)
  print(arr)
###########################################################################
root_dir = "./log/"
nm = 'lego'
#name = "dynamic_"+nm+"_1f2"
#name = 'tensorf_flower_VM'
#yuv(root_dir+name,nm)
#yuv_llff(root_dir+"llff/"+name)
#batch_yuv(root_dir,nm)


