import Codec.mergePlane as merge
import Codec.recoverPlane as recover
import Codec.mlpsaver as savemlp
import Codec.mlprecover as mlprecover
import os


## normalized planes to yuv444 frames and to video via ffmpeg

root_path = './coffee/fd1/'
root = root_path + 'nerf_synthetic/coffee_f'
scene, qp = 2, 0
dataset = 'coffee'
merge.batch_process(root, scene,root_path,'coffee')

## concat and recover planes

for idx in range(2,scene+1):
     for qp in [0,10,20,30,40]:
          yuv_files = [root+str(idx)+f"/yuv_frames/yuv_f{i}.yuv" for i in range(2,scene+1)]  # List of YUV file names
          output_video = root+str(idx)+"/yuv_frames/yuv_video_output"+str(qp)+".mp4"
          #merge.concat_yuv_to_video(yuv_files,output_video,qp)
          #recover.toframes(root+str(idx)+'/yuv_frames/yuv_video_output'+str(qp)+'.mp4',root+str(idx)+'/yuv_frames/frames_'+str(qp))
          #print("-----start replace-----")
          #recover.repalce(root_path,root,scene,qp,dataset)
          # compress and replace mlp
          
          print("==> start mlp compression")
          if qp == 0:
               mlp_yuv = [root+str(idx)+f"/yuv_mlp/mlp_f{i}.yuv" for i in range(2,scene+1)]  # List of YUV file names
               mlp_video = root+str(idx)+"/yuv_mlp/mlp_video_output.mp4"
               savemlp.concat_yuv_to_video(mlp_yuv, mlp_video)
          savemlp.batch_mlp(root,scene,root_path,qp,dataset)
          mlprecover.replace(root_path,root,scene,qp,dataset)
#recover.backtoplane(root_path,qp,scene,'worker') # for verify recovered greyscale

