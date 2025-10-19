#import ffmpeg
import os, io, glob
import torch 
import yuvio
from PIL import Image as img
import numpy as np
import imageio
import collections,pickle
import subprocess

def load_model_planes(ckpt_pth):
     checkpoint= None
     planes = {}
     if os.path.exists(ckpt_pth+'model.ckpt'):
          checkpoint = torch.load(ckpt_pth+'model.ckpt',map_location=torch.device('cpu'))
     else:
          print(ckpt_pth+'model.ckpt' +" does not exist.")
     
     #for name, tensor in checkpoint['model'].items():
               #print(f"{name}:", f"{tensor.size()}")
     
     arr = ['field.encoding.x','field.encoding.y','field.encoding.z']
     for item in arr:
          planes[item] = checkpoint['model'][item].cpu().numpy()
          #print(item, planes[item].shape)

     return planes

def grayscale(data,ph, name):
     nm = name
     
     #os.makedirs(ph+"/max_min/",exist_ok=True)

     normal = collections.OrderedDict()
     grey = collections.OrderedDict()
     
     arr = ['field.encoding.x','field.encoding.y','field.encoding.z']
     for arr_name in arr:
          
          #print("--------------"+arr_name+"-----------------")
          #normalize data
          #print(data[arr_name].squeeze().shape)
          data_min = data[arr_name].min(axis=(1,2))#, keepdims=True)
          data_max = data[arr_name].max(axis=(1,2))#, keepdims=True)
          data_norm = (data[arr_name] - data_min)/(data_max - data_min)
          normal['min_'+arr_name] = data_min[0][0]
          normal['max_'+arr_name] = data_max[0][0]

          data_grey = np.round((data_norm *255).squeeze()).astype(np.uint8)
          #print(data_grey.shape)
          grey[arr_name] = data_grey
          #print(grey[arr_name])
          #save to new file as inference 
          '''
          for channel_idx in range(ra):    #density 16
               planes = data_grey[channel_idx]
               
               plane = np.asarray(planes, dtype=np.uint8)

               plane_img = img.fromarray(plane)
               plane_img=plane_img.convert('L')
          
               plane_img.save(pth+'/'+arr_name+str(channel_idx)+".png")
          '''
     return normal,grey

     #with open(ph+"max_min/density_min.pkl","wb") as f:
          #pickle.dump(mini, f)
     #with open(ph+"max_min/density_max.pkl","wb") as f:
          #pickle.dump(maxm, f)

def yuv(greyscale, dst,idx):

     os.makedirs(dst,exist_ok=True)
     #print(dst)
     arr = ['field.encoding.x','field.encoding.y','field.encoding.z']
     y = greyscale[arr[0]]
     u = greyscale[arr[1]]
     v = greyscale[arr[2]]

     frame_444 = yuvio.frame((y, u, v), "yuv444p")
     print(f"scene_f{idx}: {dst}yuv_f{idx}.yuv")

     yuvio.imwrite(dst+'yuv_f'+str(idx)+'.yuv', frame_444)
     
     
def batch_process(root, scene, root_path, dataset,start):
     print('--> start generate frames of ' + dataset)
     #os.makedirs(ph+"/max_min/",exist_ok=True)
     min_max = collections.OrderedDict()
     
     for idx in range(start,scene+1):
          root_pth = root + str(idx) +'/Tri-MipRF/'
          for file in os.listdir(root_pth):
               print(root_pth+file+'/')
               ph = root_pth+file+'/'
               planes = load_model_planes(root_pth+file+'/')
               min_max[dataset+'_f'+str(idx)], grey= grayscale(planes,ph,dataset+'_f'+str(idx))
               dst = root_path + '/yuv_frames/compf/'
               yuv(grey,dst,idx)

     with open(root_path+"/yuv_frames/min_max.pkl","wb") as f:
          pickle.dump(min_max, f)
     

def concat_yuv_to_video(yuv_files, output_video, qp,fd, frame_rate=30, pixel_format='yuv444p'):
     resolution = '16x512'
     if fd == 3:
          resolution = '48x512'
    # Create the input string for concatenation
     concat_input = "concat:" + "|".join(yuv_files)

    # Construct the FFmpeg command
     ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-f", "rawvideo",
        "-s", resolution,  # Video resolution
        "-r", str(frame_rate),  # Frame rate
        "-pix_fmt", pixel_format,  # Pixel format
        "-i", concat_input,  # Concatenated input files
        "-c:v", "libx264",  # Video codec
        "-qp", str(qp),  # Lossless compression
        "-g", "48",  # Group of pictures
        output_video  # Output video file
     ]

     # Run the FFmpeg command
     subprocess.run(ffmpeg_command, check=True)
     print(f"Video saved as {output_video}")