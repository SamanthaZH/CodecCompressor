import Codec.mergePlane as merge
import Codec.recoverPlane as recover
import Codec.mlpsaver as savemlp
import Codec.mlprecover as mlprecover
import os


## normalized planes to yuv444 frames and to video via ffmpeg

dataset = 'cut'
fd = 3
root_path = './'+dataset+'/fd'+ str(fd)+'/'
root = root_path + 'nerf_synthetic/'+dataset+'_f'
scene, qp = 3, 0
start = 3

#merge.batch_process(root, scene,root_path,dataset,start)

## concat and recover planes
print("======>Current Scene:", scene)
for idx in range(start,scene+1):
     for qp in [0,10,20,30,40]:
          # Feature vector compression
          yuv_files = [root_path+f"yuv_frames/compf/yuv_f{i}.yuv" for i in range(start,scene+1)]  # List of YUV file names
          output_video = root_path+"yuv_frames/yuv_video_output"+str(qp)+".mp4"
          merge.concat_yuv_to_video(yuv_files,output_video,qp,fd)
          recover.toframes(root_path+'/yuv_frames/yuv_video_output'+str(qp)+'.mp4',root_path+'/yuv_frames/frames_'+str(qp),fd)
          print("-----start replace-----")
          recover.repalce(root_path,root,scene,qp,dataset,fd,start)
          
          
          # compress and replace mlp
          
          print("==> start mlp compression")
          savemlp.batch_mlp(root,scene,root_path,qp,dataset,fd)
          if qp == 0:
               mlp_yuv = [root+str(idx)+f"/yuv_mlp/mlp_f{i}.yuv" for i in range(start,scene+1)]  # List of YUV file names
               mlp_video = root+str(idx)+"/yuv_mlp/mlp_video_output.mp4"
               savemlp.concat_yuv_to_video(mlp_yuv, mlp_video,fd)
          mlprecover.replace(root_path,root,scene,qp,dataset,fd)
          
print('---Finished Compression---')
#recover.backtoplane(root_path,qp,scene,'worker') # for verify recovered greyscale

