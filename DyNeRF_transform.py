import numpy as np
import json
import cv2
import os, shutil





'''
import torch
from typing import Type
import math
from typing import List, Literal, Optional, Tuple

from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

poses_bounds = np.load('/home/jackzhu/Downloads/coffee_martini/poses_bounds.npy')
poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)  # (num_cameras, 3, 5)
heights = poses_hwf[:, 0, -1]
widths = poses_hwf[:, 1, -1]
focal = poses_hwf[:, 2, -1]
poses = poses_hwf[:, :3, :4]
poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:4]], axis=-1)
poses[..., 3] /= 5.0


f = open('/home/jackzhu/Downloads/tri-mip-dynerf/transforms_train.json')
data = json.load(f)
correct_frames = data['frames']
'''
def dataset_generation(videos,poses_bounds,type):
    poses_hwf = poses_bounds[:, :15].reshape(-1, 3, 5)  # (num_cameras, 3, 5)
    heights = poses_hwf[:, 0, -1]
    widths = poses_hwf[:, 1, -1]
    focal = poses_hwf[:, 2, -1]
    poses = poses_hwf[:, :3, :4]
    poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:4]], axis=-1)
    poses[..., 3] /= 5
    out_data = {}
    out_data['w'] = widths[0]
    out_data['h'] = heights[0]
    out_data['fl_x'] = focal[0]
    out_data['fl_y'] = focal[0]
    out_data['cx'] = widths[0]/2
    out_data['cy'] = heights[0]/2
    out_data['frames'] = []

    videos.sort()
    if type == 'train':
        for i in range(len(videos)):
            video = videos[i]
            cap = cv2.VideoCapture('/home/yl4090_1/Downloads/coffee_martini/videos/' + video)
            if cap.isOpened():
                #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames = 1
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = 0
                while frame_count < total_frames:
                    ret, frame = cap.read()
                    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    last_row = np.array([0,0,0,1])
                    transform = poses[i]
                    transform = np.vstack((transform,last_row))

                    path = 'cam'+ str(i) # + str(frame_num)

                    frame_data = {
                        'file_path': './train/'+ path,# +'.png',
                        'transform_matrix': transform.tolist(),
                        #'time': time,
                    }
                    out_data['frames'].append(frame_data)
                    alpha_channel = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                    rgba_image = np.dstack((frame, alpha_channel))

                    cv2.imwrite('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/'+'train/'+path+'.png', rgba_image)
                    #cv2.imwrite('/home/jackzhu/Downloads/nerfacto_coffee/images/' + path + '.png', frame)
                    frame_count += 1
            cap.release()
        with open('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/' + 'transforms_train.json', 'w') as out_file:
            json.dump(out_data, out_file, indent=4)
    else:
        for i in range(2):
            video = videos[i]
            cap = cv2.VideoCapture('/home/yl4090_1/Downloads/coffee_martini/videos/' + video)
            if cap.isOpened():
                # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                total_frames = 1
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = 0
                while frame_count < total_frames:
                    ret, frame = cap.read()
                    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    last_row = np.array([0, 0, 0, 1])
                    transform = poses[i]
                    transform = np.vstack((transform, last_row))

                    path = 'cam' + str(i)
                    if type == 'val':
                        frame_data = {
                            'file_path': './val/' + path,  # +'.png',
                            'transform_matrix': transform.tolist(),
                            # 'time': time,
                        }
                        out_data['frames'].append(frame_data)
                        alpha_channel = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                        rgba_image = np.dstack((frame, alpha_channel))
                        print(rgba_image)
                        cv2.imwrite('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/' + 'val/' + path + '.png', rgba_image)
                    elif type == 'test':
                        frame_data = {
                            'file_path': './test/' + path,  # +'.png',
                            'transform_matrix': transform.tolist(),
                            # 'time': time,
                        }
                        out_data['frames'].append(frame_data)
                        alpha_channel = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                        rgba_image = np.dstack((frame, alpha_channel))
                        print(rgba_image)
                        cv2.imwrite('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/' + 'test/' + path + '.png', rgba_image)
                    frame_count += 1
            cap.release()

        if type == 'val':
            with open('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/' + 'transforms_val.json', 'w') as out_file:
                json.dump(out_data, out_file, indent=4)
        elif type == 'test':
            with open('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/coffee_f1/' + 'transforms_test.json', 'w') as out_file:
                json.dump(out_data, out_file, indent=4)
        ''''''
#poses_bounds = np.load('/home/yl4090_1/Downloads/coffee_martini/poses_bounds.npy')
#videos = os.listdir("/home/yl4090_1/Downloads/coffee_martini/videos/")

#dataset_generation(videos,poses_bounds,type='train')
#dataset_generation(videos,poses_bounds,type='val')
#dataset_generation(videos,poses_bounds,type='test')


def copy_trans(src, dst):
     files = ['transforms_train.json', 'transforms_test.json','transforms_val.json']
     for item in files:
          pth = src + item
          for i in range(0,61):
               folder = 'coffee_f'+str(i)
               out = dst+folder+'/'
               print(pth,out)
               #os.remove(out+item)
               shutil.copy(pth,out)

copy_trans('./log2/','./coffee_martini/')