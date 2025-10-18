import os, subprocess, yuvio, pickle, collections
import numpy as np
import torch


def toFrames(input_file, output_dir):
     os.makedirs(output_dir, exist_ok=True)

     # FFmpeg command to generate files with sequential numbering
     ffmpeg_command = [
          "ffmpeg",
          "-y",
          "-i", input_file,
          "-s", "216x256",
          "-pix_fmt", "yuv444p",
          "-f", "segment",
          "-segment_time", "0.01",
          os.path.join(output_dir, "mlp_out_f0.yuv")
     ]

     # Run FFmpeg command
     subprocess.run(ffmpeg_command, check=True)
     
def toMLP(root_path,qp,frame, fd=1):
     #yuvs = root_path + '/yuv_mlp/mlp_'+str(qp)+'/mlp_f'+str(frame)+'.yuv'
     #yuvs = root_path + '/yuv_mlp/mlp_f'+str(frame)+'.yuv'
     toFrames(root_path + '/yuv_mlp/mlp_video_output.mp4',root_path + '/yuv_mlp/')
     yuvs = root_path + '/yuv_mlp/mlp_out_f0.yuv'
     print(yuvs)
     if fd == 1:
          frames = yuvio.imread(yuvs,256,216,'yuv444p')
     elif fd == 3:
         frames = yuvio.imread(yuvs,512,48,'yuv444p')
     
     return frames

def recover_original_data(grey, normal):
    recovered_data = collections.OrderedDict()
    
    arr = ['base', 'head']
    
    for field_name in arr:
        print(f"Recovering data for: MLP_{field_name}")

        # Load min and max values
        min_value = normal[f'min_{field_name}']
        max_value = normal[f'maxi_{field_name}']
        print(min_value,max_value)
        # Convert grayscale data back to normalized values
        if field_name == 'base':
          data_norm = grey.y.astype(np.float32) / 255.0  
          data_norm = (data_norm.flatten())[:26624]
          
        elif field_name == 'head':
          data_norm = grey.u.astype(np.float32) / 255.0
          data_norm = data_norm.reshape(-1)

        # Reverse the normalization process
        original_data = data_norm * (max_value - min_value) + min_value

        # Ensure the recovered data retains the original shape
        recovered_data[field_name] = original_data
        print(f"Recovered shape for {field_name}: {original_data.shape}")

    return recovered_data

def backtomlp(root_path,qp,s, dataset):
     #grey = toMLP(root_path,qp,str(s-2))
     grey = toMLP(root_path,qp,str(s))
     with open(root_path+'/yuv_mlp/mlp.pkl', 'rb') as file:
          data = pickle.load(file)
     print(data)
     min_max = data[dataset+'_'+str(s)]
     print(min_max)
     print('mlp_f'+str(s-2), dataset+'_f'+str(s))
     recovered = recover_original_data(grey,min_max)

     return recovered

     
def replace(root_path, root,scene, qp,dataset):
     for idx in range(2,scene+1):
          root_pth = root + str(idx) +'/Tri-MipRF/'
          for file in os.listdir(root_pth):
               print(root_pth+file+'/')
               ph = root_pth+file+'/'
               #ckpt = torch.load(ph+'model.ckpt',map_location=torch.device('cpu'))
               ckpt = torch.load(ph+'ckpt'+str(qp)+'.ckpt',map_location=torch.device('cpu'))
               
               #planes = backtomlp(root+str(idx),qp,idx,dataset)
               mlp = backtomlp(root+str(idx),qp,idx,dataset)

               arr = ['field.mlp_base.params','field.mlp_head.params']
               ckpt['model'][arr[0]] = torch.from_numpy(mlp['base'])
               ckpt['model'][arr[1]] = torch.from_numpy(mlp['head'])
               
               print(ph+'mlp'+str(qp)+'.ckpt')
               torch.save(ckpt,ph+'mlp'+str(qp)+'.ckpt')