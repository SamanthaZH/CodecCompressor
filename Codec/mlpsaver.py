import os,torch, collections, pickle
import numpy as np
from Codec.mergePlane import load_model_planes
import yuvio, subprocess


def nomalization(matrix):
     data_min = matrix.min(axis=0)
     data_max = matrix.max(axis=0)
     data_norm = (matrix-data_min)/(data_max-data_min)
     data_grey = np.round((data_norm *255).squeeze()).astype(np.uint8)
     #print(data_norm)
     return data_grey, data_min, data_max

def compress_mlp(mlp_base, mlp_head,fd):
     #print(mlp_base.shape, mlp_head.shape)
     mlp_minmax = {}
     grey_base, min_base, maxi_base = nomalization(mlp_base)
     grey_head, min_head, maxi_head = nomalization(mlp_head)
     #print("norm", grey_base, grey_head)
     #mlp_dic['mlp_base'] = grey_base
     #mlp_dic['mlp_head'] = grey_head
     if fd == 1:
          grey_base = np.pad(grey_base, (0, 216*256 - 26624), constant_values=0)
     if fd == 3:
          #print(grey_base.shape, grey_head.shape)
          grey_base = np.pad(grey_base, (0, 216*256 - 30720), constant_values=0)
     base_2d = grey_base.reshape(216,256)
     head_2d = grey_head.reshape(216,256)
     #base = (base_2d.flatten())[:26624]
     #head = head_2d.reshape(-1)
     #print(base_2d, base)
     #print(head_2d, head)
     mlp_minmax['min_base'] = min_base
     mlp_minmax['maxi_base'] = maxi_base
     mlp_minmax['min_head'] = min_head
     mlp_minmax['maxi_head'] = maxi_head
     return mlp_minmax, base_2d, head_2d

def reverse(mlp_dic):
     base_min, base_max = mlp_dic['min_base'], mlp_dic['maxi_base']
     base = mlp_dic['mlp_base'].astype(np.float32)/255
     mlp_base = base*(base_max - base_min)+base_min
     #data_norm * (max_value - min_value) + min_value
     head_min, head_max = mlp_dic['min_head'], mlp_dic['maxi_head']
     head = mlp_dic['mlp_head'].astype(np.float32)/255
     mlp_head = head*(head_max - head_min)+head_min
     return mlp_base, mlp_head

def yuv(base,head, root,idx,dataset,fd):

     dst = root + str(idx) + '/yuv_mlp/'
     os.makedirs(dst,exist_ok=True)
     #print("yuv func:",dst)
     y = base
     u = head
     v = np.zeros((216,256), dtype = np.uint8)

     frame_444 = yuvio.frame((y, u, v), "yuv444p")
     print(dst+'mlp_f'+str(idx)+'.yuv')

     yuvio.imwrite(dst+'mlp_f'+str(idx)+'.yuv', frame_444)


def concat_yuv_to_video(yuv_files, output_video,fd, frame_rate=30, pixel_format='yuv444p'):
    
     # Create the input string for concatenation
     concat_input = "concat:" + "|".join(yuv_files)
     resolution = '216x256'

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
        "-qp", "0",  # Lossless compression
        "-g", "48",  # Group of pictures
        output_video  # Output video file
     ]

     #print(f"===============================>Video saved as {output_video}")
     # Run the FFmpeg command
     subprocess.run(ffmpeg_command, check=True)
     


def batch_mlp(root, scene, root_path, qp,dataset,fd, start):
     dic_mlp =  collections.defaultdict(list)
     for idx in range(start,scene+1):
          root_pth = root + str(idx) +'/Tri-MipRF/'
          for file in os.listdir(root_pth):
               #print(root_pth+file+'/')
               ph = root_pth+file+'/'
               checkpoint=  torch.load(ph+'ckpt'+str(qp)+'.ckpt',map_location=torch.device('cpu'))
               #for name, tensor in checkpoint['model'].items():
                    #print(f"{name}:", f"{tensor.size()}")
               mlp_base = checkpoint['model']['field.mlp_base.params'].cpu().numpy()
               mlp_head = checkpoint['model']['field.mlp_head.params'].cpu().numpy()
               
               norm_mlp, base,head = compress_mlp(mlp_base, mlp_head,fd)
               #print(norm_mlp)
               dic_mlp['%s_%s'%(dataset, str(idx))] = norm_mlp

               yuv(base,head,root,idx,dataset,fd)
               
               '''
               mlp_back_base, mlp_back_head = reverse(dic_mlp['frame_%s'%str(idx)])
               checkpoint['model']['field.mlp_base.params'] = torch.from_numpy(mlp_back_base)
               checkpoint['model']['field.mlp_head.params'] = torch.from_numpy(mlp_back_head)
               torch.save(checkpoint,ph+'new_mlp.ckpt')
               '''
          #print(dic_mlp)   
     
          with open(root + str(idx) +"/yuv_mlp/mlp.pkl","wb") as f:
               pickle.dump(dic_mlp, f)
          