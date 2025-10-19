import os,torch
import pickle
from moviepy.editor import VideoFileClip
import yuvio


def addition(max_min):
  #ckpt = torch.load(new_ckpt)
  
  arr = ['app_max','app_min','density_max','density_min']
  with open(os.path.join(max_min, arr[0]+'.pkl'), 'rb') as f:
    app_max = pickle.load(f)
    #torch.save(app_max,"app_max.pkl")
  with open(os.path.join(max_min, arr[1]+'.pkl'), 'rb') as f:
    app_min = pickle.load(f)
    #torch.save(app_max,"app_min.pkl")
  with open(os.path.join(max_min, arr[2]+'.pkl'), 'rb') as f: 
    density_max = pickle.load(f)
    #torch.save(app_max,"density_max.pkl")
  with open(os.path.join(max_min, arr[3]+'.pkl'), 'rb') as f:
    density_min = pickle.load(f)
    
  additional_files = {
    "app_max":app_max,
    "app_min":app_min,
    "density_max": density_max,
    "density_min": density_min

  }
  return additional_files
  '''
  ckpt['additional_files'] = additional_files
  #make sure all necessary files are in ckpt
  print("***keys in ckpt***")
  print("ckpt key: ",ckpt.keys())
  print("******************")
  #print(ckpt['additional_files']['app_max'])
  
  torch.save(ckpt,new_ckpt)
  #print("finish saving")
  '''
  
def toempty(ckpt,new_ckpt):
  checkpoint = torch.load(ckpt)
  print(checkpoint['state_dict'].keys())
  '''
  checkpoint['state_dict']['density_plane.0'] = None
  checkpoint['state_dict']['density_plane.1'] = None
  checkpoint['state_dict']['density_plane.2'] = None
  checkpoint['state_dict']['density_line.0'] = None
  checkpoint['state_dict']['density_line.1'] = None
  checkpoint['state_dict']['density_line.2'] = None
  
  checkpoint['state_dict']['app_plane.0'] = None
  checkpoint['state_dict']['app_plane.0'] = None
  checkpoint['state_dict']['app_plane.0'] = None
  checkpoint['state_dict']['app_line.0'] = None
  checkpoint['state_dict']['app_line.0'] = None
  checkpoint['state_dict']['app_line.0'] = None
  print(checkpoint['state_dict']['density_plane.0'])
  '''
  
  item_to_remove=['density_plane.0','density_plane.1','density_plane.2','density_line.0','density_line.1','density_line.2',
          'app_plane.0','app_plane.1','app_plane.2','app_line.0','app_line.1','app_line.2',]
  for files in item_to_remove:
  #if files in your_data['checkpoint.pt']['state_dict']:
  #  del your_data['checkpoint.pt']['state_dict'][files]
    if files in checkpoint['state_dict']:
       del checkpoint['state_dict'][files]
  
  with open(new_ckpt, 'wb') as file:
    pickle.dump(checkpoint, file)
  #torch.save(checkpoint,new_ckpt)
  print("Saved successfully")


def save_yuv_pkl(root_dir,ckpt,dst,n):
  ckpt = torch.load(ckpt)
  
  
  print(ckpt.keys())
  '''
  kwargs = ckpt['kwargs']
  shape = ckpt['alphaMask.shape']
  mask = ckpt['alphaMask.mask']
  aabb = ckpt['alphaMask.aabb']
  print(ckpt['state_dict'].keys())
  '''

  #packed addition max_min
  '''
  max_min = root_dir +"max_min/"
  additional_files = addition(max_min)
  '''
  max_min = root_dir +"max_min/"
  arr = ['app_max','app_min','density_max','density_min']
  with open(os.path.join(max_min, arr[0]+'.pkl'), 'rb') as f:
    app_max = pickle.load(f)
    #torch.save(app_max,"app_max.pkl")
  with open(os.path.join(max_min, arr[1]+'.pkl'), 'rb') as f:
    app_min = pickle.load(f)
    #torch.save(app_max,"app_min.pkl")
  with open(os.path.join(max_min, arr[2]+'.pkl'), 'rb') as f: 
    density_max = pickle.load(f)
    #torch.save(app_max,"density_max.pkl")
  with open(os.path.join(max_min, arr[3]+'.pkl'), 'rb') as f:
    density_min = pickle.load(f)
  
  ckpt['additional_files'] = {
    "app_max":app_max,
    "app_min":app_min,
    "density_max": density_max,
    "density_min": density_min
  }
  
  
  #packed videos
  density_video_file_path = root_dir+ 'yuv/density_plane/density_plane_'+n+'.mp4'
  app_video_file_path = root_dir+'yuv/app_plane/app_plane_'+n+'.mp4'
  with open(density_video_file_path, 'rb') as video_file:
    density_video_data = video_file.read()
  with open(app_video_file_path, 'rb') as video_file:
    app_video_data = video_file.read()
  # Create a Python object and include the video data
  #density_video_data = VideoFileClip(density_video_file_path)
  #app_video_data = VideoFileClip(app_video_file_path)
  video_container = {
    'density_video': density_video_data,
    #'density_video_filename': 'density.mp4',
    'app_video': app_video_data,
    #'app_video_filename': 'app.mp4'
    }


  data_dict={
    'checkpoint': ckpt,
    #'addition': additional_files,
    'videos':video_container
  }
  for keys in data_dict:
    print(keys)
  
  
  with open(dst, 'wb') as f:
    pickle.dump(data_dict, f)
  print("Saved")
  

'''
pth = "./log/tensorf_lego_VM_192_org/"
ckpt = "ckpt_lego_VM_192.th"
new_ckpt = "ckpt_lego_VM_192_new.th"
dst = "ckpt_lego_VM_192_yuv_10.pkl"
toempty(pth+ckpt,pth+new_ckpt)
save_yuv_pkl(pth+new_ckpt,pth+dst)
'''
n =str(35)
root_dir = "./log/mm2023/tensorf_lego_VM_192_org/"
ckpt = root_dir+"modified_file.th"
dst = root_dir+"ckpt_lego192_"+n+".pkl"
save_yuv_pkl(root_dir,ckpt,dst,n)
#toempty(ckpt,dst)
