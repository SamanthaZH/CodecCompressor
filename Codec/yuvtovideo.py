import os
import imageio
import io
import yuvio

def yuv2video_test(pth):

  for files in os.listdir(pth):
      print(pth+"/"+files)
      reader = yuvio.get_reader("example_yuv444.yuv", 453, 257, "yuv444p")
      writer = yuvio.get_writer("example_yuv444_copy.yuv", 453, 257, "yuv444p")


def yuv2video(pth):
  arr_shape = {"density_plane-0":(453,257),
                   "density_plane-1": (231,257),
                   "density_plane-2": (231,453),
                   "density_line-0": (231,1),
                   "density_line-1": (453,1),
                   "density_line-2": (257,1),
                   "app_plane-0": (453,257),
                   "app_plane-1": (231,257),
                   "app_plane-2": (231,453),
                   "app_line-0": (231,1),
                   "app_line-1": (453,1),
                   "app_line-2": (257,1)
                   }
  
  for folder in os.listdir(pth):
    for files in os.listdir(pth+folder):
      print(pth+folder+"/"+files)
      
      reader = yuvio.get_reader("example_yuv420p.yuv", 1920, 1080, "yuv420p")
      
yuv2video("./log/tensorf_lego_VM_192/yuv/app_plane-0")
#yuv2video("./log/tensorf_lego_VM_192/yuv/")