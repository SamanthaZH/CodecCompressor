import os, subprocess, yuvio, pickle, collections
import numpy as np
import torch


def toframes(input_file, output_dir,fd):
     resolution = '16x512'
     if fd == 3:
          resolution = "48x512"
     os.makedirs(output_dir, exist_ok=True)

     # FFmpeg command to generate files with sequential numbering
     ffmpeg_command = [
          "ffmpeg",
          "-y",
          "-i", input_file,
          "-s", resolution,
          "-pix_fmt", "yuv444p",
          "-f", "segment",
          "-segment_time", "0.01",
          os.path.join(output_dir, "yuv_f%d.yuv")
     ]

     # Run FFmpeg command
     subprocess.run(ffmpeg_command, check=True)
     
def toPlanes(root_path,qp,frame,fd):
     #print(frame)
     yuvs = root_path + '/yuv_frames/frames_'+str(qp)+'/yuv_f'+str(frame)+'.yuv'
     print(frame, yuvs)
     frames = yuvio.imread(yuvs,512,16,'yuv444p')
     if fd == 3:
          frames = yuvio.imread(yuvs,512,48,'yuv444p')
     
     return frames

def recover_original_data(grey, normal):
    recovered_data = collections.OrderedDict()
    
    arr = ['field.encoding.x','field.encoding.y','field.encoding.z']
    
    for field_name in arr:
        print(f"Recovering data for: {field_name}")

        # Load min and max values
        min_value = normal[f'min_{field_name}']
        max_value = normal[f'max_{field_name}']
        
        # Convert grayscale data back to normalized values
        if field_name[-1] == 'x':
          data_norm = grey.y.astype(np.float32) / 255.0  
        elif field_name[-1] == 'y':
          data_norm = grey.u.astype(np.float32) / 255.0  
        elif field_name[-1] == 'z':
          data_norm = grey.v.astype(np.float32) / 255.0  
        #print(grey.y, grey.u, grey.v)

        # Reverse the normalization process
        original_data = data_norm * (max_value - min_value) + min_value

        original_data = np.expand_dims(original_data, axis=(0, -1))

        # Ensure the recovered data retains the original shape
        recovered_data[field_name] = original_data
        print(f"Recovered shape for {field_name}: {original_data.shape}")

    return recovered_data

def backtoplane(root,root_path,qp,s, dataset,start,fd):
     grey = toPlanes(root_path,qp,str(s-start),fd) # for multiple scene
     #grey = toPlanes(root_path,qp,0,fd) # for test one scene
     with open(root_path+'/yuv_frames/min_max.pkl', 'rb') as file:
          data = pickle.load(file)
     #print(data)
     min_max = data[dataset+'_f'+str(s)]
     #print(min_max)
     print('yuv_f'+str(s-start), dataset+'_f'+str(s))
     recovered = recover_original_data(grey,min_max)

     return recovered
   
def repalce(root_path, root,scene, qp,dataset,fd,start,idx):
     #for idx in range(start,scene+1):
          root_pth = root + str(idx) +'/Tri-MipRF/'
          for file in os.listdir(root_pth):
               print(root_pth+file+'/')
               ph = root_pth+file+'/'
               ckpt = torch.load(ph+'model.ckpt',map_location=torch.device('cpu'))
               
               planes = backtoplane(root+str(idx),root_path,qp,idx,dataset,start,fd)

               arr = ['field.encoding.x','field.encoding.y','field.encoding.z']
               ckpt['model'][arr[0]] = torch.from_numpy(planes[arr[0]])
               ckpt['model'][arr[1]] = torch.from_numpy(planes[arr[1]])
               ckpt['model'][arr[2]] = torch.from_numpy(planes[arr[2]])

               torch.save(ckpt,ph+'ckpt'+str(qp)+'.ckpt')
               print("----------update ckpt saved to " +ph+'ckpt'+str(qp)+'.ckpt' )



