#yuv_read is read yuv files saved back to yuv files, this is all frames
# to make it back to metrics. 
import cv2
from PIL import Image as img
import numpy as np
import os,pickle
import imageio
import io
import yuvio
import glob
import collections
import torch



def den(nm):
  density_plane_0 = np.zeros((16, 430, 245), dtype=np.uint8)
  density_plane_1 = np.zeros((16, 255, 245), dtype=np.uint8)
  density_plane_2 = np.zeros((16, 255, 430), dtype=np.uint8)
  if nm == 'lego': 
    density_plane_0 = np.zeros((16, 420, 237), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 270, 237), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 270, 420), dtype=np.uint8)
  if nm == 'hotdog':
    density_plane_0 = np.zeros((16, 467, 463), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 124, 463), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 124, 467), dtype=np.uint8)
  if nm == "drum":
    density_plane_0 = np.zeros((16, 289, 378), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 246, 378), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 246, 289), dtype=np.uint8)
  if nm == "mic":
    density_plane_0 = np.zeros((16, 303, 306), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 289, 306), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 289, 303), dtype=np.uint8)
  if nm == "ship":
    density_plane_0 = np.zeros((16, 365, 368), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 200, 368), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 200, 365), dtype=np.uint8)
  if nm == "materials":
    density_plane_0 = np.zeros((16, 421, 531), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 120, 531), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 120, 421), dtype=np.uint8)
  if nm == "ficus":
    density_plane_0 = np.zeros((16, 294, 207), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 442, 207), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 442, 294), dtype=np.uint8)
  if nm =='chair':
    density_plane_0 = np.zeros((16, 270, 257), dtype=np.uint8)
    density_plane_1 = np.zeros((16, 388, 257), dtype=np.uint8)
    density_plane_2 = np.zeros((16, 388, 270), dtype=np.uint8)
  return density_plane_0, density_plane_1,density_plane_2

def app(nm):
  app_plane_0 = np.zeros((48, 430, 245), dtype=np.uint8)
  app_plane_1 = np.zeros((48, 255, 245), dtype=np.uint8)
  app_plane_2 = np.zeros((48, 255, 430), dtype=np.uint8)
  if nm == 'lego':
    app_plane_0 = np.zeros((48, 420, 237), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 270, 237), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 270, 420), dtype=np.uint8)
  if nm == 'hotdog':
    app_plane_0 = np.zeros((48, 467, 463), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 124, 463), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 124, 467), dtype=np.uint8)
  if nm == 'drum':
    app_plane_0 = np.zeros((48, 289, 378), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 246, 378), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 246, 289), dtype=np.uint8)
  if nm == "mic":
    app_plane_0 = np.zeros((48, 303, 306), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 289, 306), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 289, 303), dtype=np.uint8)
  if nm == "ship":
    app_plane_0 = np.zeros((48, 365, 368), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 200, 368), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 200, 365), dtype=np.uint8)
  if nm == "materials":
    app_plane_0 = np.zeros((48, 421, 531), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 120, 531), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 120, 421), dtype=np.uint8)
  if nm == "ficus":
    app_plane_0 = np.zeros((48, 294, 207), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 442, 207), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 442, 294), dtype=np.uint8)
  if nm =='chair':
    app_plane_0 = np.zeros((48, 270, 257), dtype=np.uint8)
    app_plane_1 = np.zeros((48, 388, 257), dtype=np.uint8)
    app_plane_2 = np.zeros((48, 388, 270), dtype=np.uint8)
  return app_plane_0,app_plane_1,app_plane_2 

def yuv_read(nm,width,height, src,src2,dst,size):
  #read yuv444 files(extract from video) and save back as npy
  os.makedirs(dst,exist_ok=True)
  #folder_path = src

  #den_size = [(16,) + sublist for sublist in size]
  #app_size = [(48,) + sublist for sublist in size]
  #print(den_size[0],app_size[1])
  file_pattern = "*.yuv"
  file_paths = glob.glob(src + "/" + file_pattern)
  file_paths2 = glob.glob(src2 + "/" + file_pattern)
  
  density_plane_0 = np.zeros((16, 378, 284), dtype=np.uint8)
  density_plane_1 = np.zeros((16, 250, 284), dtype=np.uint8)
  density_plane_2 = np.zeros((16, 250, 378), dtype=np.uint8)
  #density_plane_0 = np.zeros(den_size[0], dtype=np.uint8)
  #density_plane_1 = np.zeros(den_size[1], dtype=np.uint8)
  #density_plane_2 = np.zeros(den_size[2], dtype=np.uint8)

  #density_plane_0,density_plane_1,density_plane_2 =den(nm)
  
  i = 0
  j = 0
  for i in range(0,16):
    filename = "density_plane-"+str(i)+".yuv"
    #print(filename)
    
    frames = yuvio.imread(src+filename,width,height,'yuv444p')
    
    
    #y = frames.y[:size[0][0],:size[0][1]]
    #u = frames.u[:size[1][0],:size[1][1]]
    #v = frames.v[:size[2][0],:size[2][1]]
    
    y = frames.y[:378,:284]
    
    u = frames.u[:250, :284]
    
    v = frames.v[:250, :378]
    
    if nm == 'lego':
      y = frames.y[:420,:237]
      u = frames.u[:270,:237]
      v = frames.v[:270,:420]
    if nm == "hotdog":
      y = frames.y[:467,:463]
      u = frames.u[:124,:463]
      v = frames.v[:124,:467]
    if nm == 'drum':
      y = frames.y[:289, :378]
      u = frames.u[:246, :378]
      v = frames.v[:246, :289]
    if nm == 'mic':
      y = frames.y[:303, :306]
      u = frames.u[:289, :306]
      v = frames.v[:289, :303]
    if nm == 'ship':
      y = frames.y[:365, :368]
      u = frames.u[:200, :368]
      v = frames.v[:200, :365]
    if nm == 'materials':
      y = frames.y[:421, :531]
      u = frames.u[:120, :531]
      v = frames.v[:120, :421]
    if nm == 'ficus':
      y = frames.y[:294, :207]
      u = frames.u[:442, :207]
      v = frames.v[:442, :294]
    if nm == 'chair':
      y = frames.y[:270, :257]
      u = frames.u[:388, :257]
      v = frames.v[:388, :270]
    
    
    density_plane_0[i]= y
    density_plane_1[i] = u
    density_plane_2[i] = v
    i +=1
  print(density_plane_0.shape)
  np.save(dst+"density_plane-0",density_plane_0)
  np.save(dst+"density_plane-1",density_plane_1)
  np.save(dst+"density_plane-2",density_plane_2)


  # -------appearance component----------
  app_plane_0 = np.zeros((48, 378, 284), dtype=np.uint8)
  app_plane_1 = np.zeros((48, 250, 284), dtype=np.uint8)
  app_plane_2 = np.zeros((48, 250, 378), dtype=np.uint8)
  #app_plane_0 = np.zeros(app_size[0], dtype=np.uint8)
  #app_plane_1 = np.zeros(app_size[1], dtype=np.uint8)
  #app_plane_2 = np.zeros(app_size[2], dtype=np.uint8)
  
  #app_plane_0,app_plane_1,app_plane_2 =app(nm)
  
  for i in range(0,48):
    filename = "app_plane-"+str(i)+".yuv"
    #print(filename)
    #width,height = 420,420 #432,432
    #height = 432
    frames = yuvio.imread(src2+filename,width,height,'yuv444p')
    #y = frames.y[:size[0][0],:size[0][1]]
    #u = frames.u[:size[1][0],:size[1][1]]
    #v = frames.v[:size[2][0],:size[2][1]]
    
    y = frames.y[:378,:284]
    u = frames.u[:250, :284]
    v = frames.v[:250, :378]
    if nm == 'lego':
      y = frames.y[:420,:237]
      u = frames.u[:270,:237]
      v = frames.v[:270, :420]
    if nm == "hotdog":
      y = frames.y[:467,:463]
      u = frames.u[:124,:463]
      v = frames.v[:124,:467]
    if nm == 'drum':
      y = frames.y[:289, :378]
      u = frames.u[:246, :378]
      v = frames.v[:246, :289]
    if nm == 'mic':
      y = frames.y[:303, :306]
      u = frames.u[:289, :306]
      v = frames.v[:289, :303]
    if nm == 'ship':
      y = frames.y[:365, :368]
      u = frames.u[:200, :368]
      v = frames.v[:200, :365]
    if nm == 'materials':
      y = frames.y[:421, :531]
      u = frames.u[:120, :531]
      v = frames.v[:120, :421]
    if nm == 'ficus':
      y = frames.y[:294, :207]
      u = frames.u[:442, :207]
      v = frames.v[:442, :294]
    if nm == 'chair':
      y = frames.y[:270, :257]
      u = frames.u[:388, :257]
      v = frames.v[:388, :270]

    app_plane_0[i]= y
    app_plane_1[i] = u
    app_plane_2[i] = v
    j +=1
  print(app_plane_2.shape)
  np.save(dst+"app_plane-0",app_plane_0)
  np.save(dst+"app_plane-1",app_plane_1)
  np.save(dst+"app_plane-2",app_plane_2)

def llff_read(nm,width,height,src,src2,dst):
  #read yuv444 files(extract from video) and save back as npy
  os.makedirs(dst,exist_ok=True)
  #folder_path = src
  file_pattern = "*.yuv"
  file_paths = glob.glob(src + "/" + file_pattern)
  file_paths2 = glob.glob(src2 + "/" + file_pattern)
  
  density_plane_0 = np.zeros((16, 786, 706), dtype=np.uint8)
  density_plane_1 = np.zeros((4, 471, 706), dtype=np.uint8)
  density_plane_2 = np.zeros((4, 471, 786), dtype=np.uint8)
  
  #if nm == "trex" or "flower"or 'fern':
  if nm in ['trex','flower','fern','leaves']:
    density_plane_0 = np.zeros((16, 786, 706), dtype=np.uint8)
    density_plane_1 = np.zeros((4, 471, 706), dtype=np.uint8)
    density_plane_2 = np.zeros((4, 471, 786), dtype=np.uint8)

  i = 0
  j = 0
  
  for i in range(4):
    
    filename = "density_plane-"+str(i)+".yuv"
    #print(filename)
    
    frames = yuvio.imread(src+filename,width,height,'yuv444p')
    y = frames.y[:786,:706]
    u = frames.u[:471, :706]
    v = frames.v[:471, :786]
    if nm =='trex' or 'flower' or 'fern' or 'leaves':
      y = frames.y[:786,:706]
      #print(y)
      u = frames.u[:471, :706]
      #print(u)
      v = frames.v[:471, :786]

      density_plane_0[i]= y
      density_plane_1[i] = u
      density_plane_2[i] = v
      i +=1
    #print(density_plane_0.shape)
    np.save(dst+"density_plane-0",density_plane_0)
    np.save(dst+"density_plane-1",density_plane_1)
    np.save(dst+"density_plane-2",density_plane_2)
  for i in range(4,16):
    
    filename = "density_plane-"+str(i)+".yuv"
    #print(filename)
    
    frames = yuvio.imread(src+filename,width,height,'yuv444p')
    y = frames.y[:786,:706]
    if nm =='trex' or 'flower'or 'fern'or 'leaves':
      y = frames.y[:786,:706]

      density_plane_0[i]= y
      i +=1
    #print(density_plane_0.shape)
    np.save(dst+"density_plane-0",density_plane_0)


  
  app_plane_0 = np.zeros((48, 786, 706), dtype=np.uint8)
  app_plane_1 = np.zeros((12, 471, 786), dtype=np.uint8)
  app_plane_2 = np.zeros((12, 471, 706), dtype=np.uint8)
  if nm == "trex" or "flower" or 'fern'or 'leaves':
    app_plane_0 = np.zeros((48, 786, 706), dtype=np.uint8)
    app_plane_1 = np.zeros((12, 471, 706), dtype=np.uint8)
    app_plane_2 = np.zeros((12, 471, 786), dtype=np.uint8)
  
  
  for i in range(12):
    filename = "app_plane-"+str(i)+".yuv"
    #print(filename)
    #width,height = 800,800 #432,432
    #height = 432
    frames = yuvio.imread(src2+filename,width,height,'yuv444p')
    
    y = frames.y[:786,:706]
    #print(y)
    u = frames.u[:471, :706]
    #print(u)
    v = frames.v[:471, :786]
    if nm =="trex" or "flower"or 'fern'or 'leaves':
      y = frames.y[:786, :706]
      u = frames.u[:471, :706]
      v = frames.v[:471, :786]

    app_plane_0[i]= y
    app_plane_1[i] = u
    app_plane_2[i] = v
    j +=1
  print(app_plane_2.shape)
  np.save(dst+"app_plane-0",app_plane_0)
  np.save(dst+"app_plane-1",app_plane_1)
  np.save(dst+"app_plane-2",app_plane_2)

  for i in range(12,48):
    filename = "app_plane-"+str(i)+".yuv"
    #print(filename)
    #width,height = 800,800 #432,432
    #height = 432
    frames = yuvio.imread(src2+filename,width,height,'yuv444p')
    
    y = frames.y[:786,:706]
  
    if nm =="trex" or "flower"or 'fern'or 'leaves':
      y = frames.y[:786, :706]

    app_plane_0[i]= y
    j +=1
  
  np.save(dst+"app_plane-0",app_plane_0)

def batch_yuvframe(root_dir,width,height,size,nm):
  with open(size, 'rb') as f:
    my_dict = pickle.load(f)
  print(my_dict)

  for i in range(1,2):
    name = "dynamic_"+nm+"_"+str(i)+"f"+str(i+1)
    pth = root_dir+name+'/'
    print("=============="+name+"==================")
    #old_ckpt = pth + name+".th"
    new_ckpt = pth + "ckpt_"+name+".th"
    print(new_ckpt)
    sz = my_dict[name]
    print(sz)
    qp= [0,10,20,30,40]
    #qp=[0,10]
    for j in qp:
        j = str(j)
        yuv_read(name,width,height,pth+"/yuv/density_plane/frames_"+j+'/',pth+"/yuv/app_plane/frames_"+j+'/',pth+"/yuv/ckpt"+j+'/',sz)

        replace(pth+"/yuv/ckpt"+j+'/',new_ckpt,pth+"ckpt_"+name+"_yuv"+j+".th")
   
def replace(new_paras,old_ckpt,new_ckpt):

  print("src: ",new_paras)
  #quantized parameters
  dic = collections.OrderedDict()
  for files in os.listdir(new_paras):
    #print(files.split("."))
    print(files)
    name = files.split('.')[0].replace("-",".")
    #print(name)
    file = np.load(new_paras+files)
    print(file.dtype) 
    file_tensor = torch.tensor(file)
    if "plane" in name:
      file_tensor = file_tensor.unsqueeze(dim=0)
    if "line" in name:
      file_tensor = file_tensor.unsqueeze(dim=0)
    
    dic[name] = file_tensor
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = torch.load(old_ckpt, map_location=device)
  
  print(dic.keys())

  for idd in dic.keys():
    #print(idd)
    '''
    if "plane" in idd:
      ckpt['state_dict'][idd]=dic[idd]
      print(idd)
    '''
    #dic[idd] = dic[idd]/255
    #dic[idd] = dic[idd].to(torch.uint8)
    #print(dic[idd])
    #print(np.min(dic[idd]))
    #print(np.max(dic[idd]))
  
    ckpt['state_dict'][idd] = dic[idd]

  print(ckpt['state_dict'].keys())

  #------Save quantized checkpoint --------
  print(new_ckpt)
  torch.save(ckpt, new_ckpt)
  print("saved quantized ckeckpoint")

   

#-------------------------------------------------------
#hotdog 467; chair388 lego 420; drum: 378 mic:306 ship:368 materials:531 ficus:442
#trex: 786 dynamic_lego:480
#----------Static scene---------
name = 'flower'

width,height = 786,786
r = str(40)
n = r+"/"
pth = "./log/llff/tensorf_"+name+"_VM/"
old_ckpt = pth + name+".th"
new_ckpt = pth + "ckpt_"+name+"_VM.th"

max_min = pth +"max_min/"

size = pth+"data.pkl"


#yuv_read(name,width,height,pth+"/yuv/density_plane/frames_"+n,pth+"/yuv/app_plane/frames_"+n,pth+"/yuv/ckpt"+n,size)
llff_read(name,width,height,pth+"/yuv/density_plane/frames_"+n,pth+"/yuv/app_plane/frames_"+n,pth+"/yuv/ckpt"+n)

print("*********************read frames to npy*************************")
#addition(max_min,old_ckpt,new_ckpt)
print("********************add addtitional***************")
#replace(new_paras,old_ckpt,new_ckpt)
replace(pth+"/yuv/ckpt"+n,new_ckpt,pth+"ckpt_"+name+"_yuv"+r+".th")
#print("----------saved new checkpoint---------------------")
'''

# -------------Dynamic scene--------------
root_dir="./log/pig/"
size = root_dir+"data.pkl"
#batch_yuvframe(root_dir,400,400,size,"pig")
'''