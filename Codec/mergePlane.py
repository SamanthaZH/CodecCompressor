#import ffmpeg
import os
import torch 
import yuvio


def load_model(ckpt_pth):
     checkpoint= None
     if os.path.exists(ckpt_pth+'model.ckpt'):
          checkpoint = torch.load(ckpt_pth+'model.ckpt',map_location=torch.device('cpu'),weights_only=True)

          for name, tensor in checkpoint['model'].items():
               print(f"{name}:", f"{tensor.size()}")
     else:
          print(ckpt_pth+'model.ckpt' +" does not exist.")

     return checkpoint
