# Codec Folder contains all compression code, including patch vector to video and recovery
Check compression, in codec.py
'python codec.py'

Check the result
run 'python test_model.py --ginc config_files/blender/test_only_f5.gin' 
It will auto run all compressed, the model named "ckpt#.ckpt" is Feature compression only, the model named "mlp#.chpt" is compressed both feature vector and MLP.
Change dataset name and model dir before run it

The Sample result of cut_steak:
```
                        qp0              qp10             qp20              qp30           qp40 
feature_only
FD1:       f2     27.6388	0.9589   27.6266	0.9589   26.9128	0.9582	 24.5163	0.9555	  18.669	0.9511
           f3     27.985	0.9588	 27.8272	0.9586	 27.1293	0.9574	 23.8661	0.9546	  12.569	0.9420
FD3:              30.7436	0.9711	 30.6358	0.9710	  29.8469	0.9701	 12.8522	0.9448	  8.1900	0.9338	
                  30.9104	0.9710	 30.7771	0.9708	 29.1763	0.9589	 14.2161	0.9463	  8.6474	0.9333
                  
feature&mlp  
FD1:              27.0126	0.9585	 26.9222	0.9584	 26.4088	0.9580	 24.2634	0.9555	  18.5187	0.9513
                  25.2947	0.9578	 25.5554	0.9577	 24.9602	0.9566	 23.4309	0.9540	  12.6449	0.9414
FD3:              28.4240	0.9698	 28.4557	0.9698	 27.7249	0.9693	 12.1206	0.9435	  7.6720  0.9328
                  30.5049	0.9704	 30.4146	0.9703	 29.0037	0.9683	 14.403	  0.9477	  8.8251	0.9349
```

Example model size per scene:
```
FD1: 10.41MB                                                              single scene                              two scene
compresed feature plane + normpare_fp + compressed mlp + normpare_mlp =  20.06KB + 375B + 67.56KB + 270B         33.82KB + 620B
                                                                         16.44KB                                 25.28KB
                                                                         9.78KB                                  15.40KB
                                                                         5.62KB                                  8.80KB
                                                                         3.65KB                                  4.99KB

FD3: 10.61MB
compresed feature plane + normpare_fp + compressed mlp + normpare_mlp =  55.47KB + 375B + 72.87KB + 270B        100.34KB + 620B
                                                                         46.05KB                                80.65KB
                                                                         27.66KB                                48.08KB
                                                                         14.19KB                                24.62KB 
                                                                         6.16KB                                 9.35KB

Approximate 2.43KB if not include occ(8 MB), binary, feature_planes and mlp
```                                                                         
Note: Shared occupancy for all scenes, binary removed(does not affect result)
Note: compressed result difference, but noT too much like cut_steak is 20.06, sear is 19.09, coffee is 19.47

# Data Preparation
Take our generated lego dataset as example, put `rename.py` under the lego directory
```
lego
├── train_80
├── val_20
├── rename.py
```

Create two folders, `train` and `val`, modify the frame index in `rename.py` to copy corresponding frame to the folder


`modify_json.py`

Put the orginal `transform_xx.json` file together and run `modify_json.py` to generate new json file with `new_transform_xx.json`, then put the `train`, `val` and `new_transform_xx.json` into the project folder, `Tri-MipRF-CP/nerf_synthetic/lego_fx/`, rename `new_transform_xx.json` as `transform_xx.json`

The `new_transform_xx.json` can be used for any frame index

# Training
## Train Tri-MipRF for Frame 1
`python main.py --ginc config_files/blender/f1.gin`

Note the following configs:
```
RayDataset.scene = 'lego_f1'

Trainer.is_first_frame = True

TriMipEncoding.include_cp = False
```

The trained model will be saved at: `log2/nerf_synthetic/lego_f1/Tri-MipRF/$date_$time`, 
where `$date` and `$time` are when the training was started. 

## Train follow-up frames using the triplane feature map of Frame 1
These follow-up frames are represented as triplane feature map of Frame 1 and cp features for the frame itself (plus update occupancy grid and MLP weights)

- Inside `log2/nerf_synthetic/lego_f1/Tri-MipRF/`, create a new folder, `model_w_ds`
- Copy `log2/nerf_synthetic/lego_f1/Tri-MipRF/$date_$time/fm.ckpt` into this folder. This `fm.ckpt` stores the triplane feature map of Frame 1 that we want to use for all follow-up frames. 
- Run `python main.py --ginc config_files/blender/f$n.gin`, where `$n` is the frame number.
- The results and model will be saved at: `log2/nerf_synthetic/lego_f$n/Tri-MipRF/$date_$time`. Here, `model.ckpt` should not include the triplane features (i.e., saved as `fm.ckpt` of frame 1).

An example config is at: `config_files/blender/f5.gin`. Note the following configs:
```
RayDataset.scene = 'lego_f5'

Trainer.is_first_frame = False

TriMipEncoding.include_cp = True
TriMipEncoding.comb = 5
TriMipEncoding.feature_dim_factor = 1
```
Set `TriMipEncoding.feature_dim_factor` to larger than 1 can improve the PSNR results, but increase representation size. (See below)

# Model Details
## Five combinations of CP feature and FM feature
1. CP feature add to the original triplane feature (not that good)
2. CP feature multiply the original triplane feature (not that good)
3. CP feature product and then concatenate to the original triplane feature
4. CP feature directly concatenate to the original triplane feature
5. **Calculate the product of CP features and concatenate with the triplane features (the best so far)**

Inside `class TriMipEncoding(nn.Module):`
```python
# set which of the 5 combinations to use
self.comb = comb

# set the feature length factor of the CP feature compared to triplane feature,
# e.g., a factor of 3 means the CP feature length is 3 times the triplane feature length
# A larger feature dim factor can result in better results, but at the cost of larger model size
self.feature_dim_factor = feature_dim_factor
```
These settings are configured by `gin` config. 

# Evaluation
Use `test_model.py` for evaluation only. 
## Evaluating Frame 1
Run `python test_model.py --ginc config_files/blender/test_only_f1.gin`
Note the following configuration in `test_only_f1.gin`
```python
main.saved_model_dir = '2024-07-10_15-46-34' # make sure it belongs to the correct scene
# the actual path is RayDataset.scene_type/RayDataset.scene/main.model_name/main.saved_model_dir
# e.g., nerf_synthetic/lego_f1/Tri-MipRF/2024-07-10_15-46-34

RayDataset.scene = 'lego_f1' # must be updated to the correct scene name
```

## Evaluating other frames with CP features
Run `python test_model.py --ginc config_files/blender/test_only_f1.gin`
Note the following configuration in `test_only_f1.gin`
```python
main.saved_model_dir = '2024-07-10_15-32-51' # make sure it belongs to the correct scene
# the actual path is RayDataset.scene_type/RayDataset.scene/main.model_name/main.saved_model_dir
# e.g., nerf_synthetic/lego_f5/Tri-MipRF/2024-07-10_15-32-51

RayDataset.scene = 'lego_f5' # must be updated to the correct scene name
```

In this case, `fm.ckpt` for the triplane features will be read from `model_w_ds` folder under the first frame.

`model.ckpt`, containing the CP features, MLP, and occupancy grid, will be read from `main.saved_model_dir` folder under the corresponding frame.

# Results before deferred shading

On a 3080Ti machine:

data(lego f1 and f5), model and render results: https://drive.google.com/drive/folders/1DoseZgZ-Gu3xmrTUfbXNXHc_XWUsDljR?usp=sharing


tri_mip_f1.py
the impact of feature dim and plane size on lego frame1(codepoint: neural_field/field/trimipRF.py line 18 and 19)
- f16 p512: 38.67/58MB
- f8 p128: 33.2/11.8MB
- f4 p64: 28.56/10.5
- f4 p32: 27/10.34

CP feature with different combination on lego frame5:


- train with tri_mip_f1.py: 38.5/58MB
- eval with f1 model: 22


ray_sample: 512(codepoint: neural_field/model/trimipRF.py line19)
- comb1/25k: 31.75/10.4MB
- comb1/50k: 32.22/10.4MB
- comb2/50k: 30.29/10.4MB
- comb3/50k: 32.98/10.4MB
- comb4/50k: 33.18/10.4MB


ray_sample: 1024
- comb3/50k: 34.25/10.4MB
- comb4/50k: OOM
