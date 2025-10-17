# Codec Folder contains all compression code, including patch vector to video and recovery

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
