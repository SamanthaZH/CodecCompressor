import Codec.mergePlane as merge
import Codec.recoverPlane as recover
import Codec.mlpsaver as savemlp
import Codec.mlprecover as mlprecover
import os,pickle


## normalized planes to yuv444 frames and to video via ffmpeg

'''
dataset = 'lego'
fd = 1
#root_path = './'+dataset+'/fd'+ str(fd)+'/'
root_path = './log_sample/fd'+ str(fd)+'/'
root = root_path + 'nerf_synthetic/'+dataset+'_f'
scene, qp = 2, 0
start = 2
'''
'''
dataset = 'sear'
fd = 1
root_path = './'+dataset+'/fd'+ str(fd)+'/'
#root_path = './log_sample/fd'+ str(fd)+'/'
root = root_path + 'nerf_synthetic/'+dataset+'_f'
scene, qp = 3, 0
start = 2
'''
# concat and recover planes
merge.batch_process(root, scene,root_path,dataset,start)
for qp in [0,10,20,30,40]:
          # Feature vector compression
          yuv_files = [root_path+f"yuv_frames/compf/yuv_f{i}.yuv" for i in range(start,scene+1)]  # List of YUV file names
          output_video = root_path+"yuv_frames/yuv_video_output"+str(qp)+".mp4"
          merge.concat_yuv_to_video(yuv_files,output_video,qp,fd)
for idx in range(start,scene+1):
     for qp in [0,10,20,30,40]:
          print("======>Current Scene:", idx, "QP:", qp)
          # Feature vector compression
          #yuv_files = [root_path+f"yuv_frames/compf/yuv_f{i}.yuv" for i in range(start,scene+1)]  # List of YUV file names
          #output_video = root_path+"yuv_frames/yuv_video_output"+str(qp)+".mp4"
          #merge.concat_yuv_to_video(yuv_files,output_video,qp,fd)
          recover.toframes(root_path+'/yuv_frames/yuv_video_output'+str(qp)+'.mp4',root_path+'/yuv_frames/frames_'+str(qp),fd)
          print("-----start replace-----")
          recover.repalce(root_path,root,scene,qp,dataset,fd,start,idx)
         
          # compress and replace mlp
for idx in range(start,scene+1):
     for qp in [0,10,20,30,40]:          
          print("==> start mlp compression",idx, qp)
          #savemlp.batch_mlp(root,scene,root_path,qp,dataset,fd,start)
          if qp == 0:
               savemlp.batch_mlp(root,scene,root_path,qp,dataset,fd,start)
               mlp_yuv = [root+str(idx)+f"/yuv_mlp/mlp_f{idx}.yuv"] #for i in range(start,scene+1)]  # List of YUV file names
               mlp_video = root+str(idx)+"/yuv_mlp/mlp_video_output.mp4"
               print("=====================>",mlp_yuv, mlp_video)
               savemlp.concat_yuv_to_video(mlp_yuv, mlp_video,fd)
               print(f"===============================>Video saved as {mlp_video}")
          mlprecover.replace(root_path,root,scene,qp,dataset,fd,start,idx)
          
print('---Finished Compression---')
#recover.backtoplane(root_path,qp,scene,'worker') # for verify recovered greyscale

