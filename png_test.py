import cv2
import numpy as np
import torch
import json

#img_lego = cv2.imread('/home/yl4090_1/Tri-MipRF-CP/nerf_synthetic/lego_f20/train/r_0.png',cv2.IMREAD_UNCHANGED)

#img_dy = cv2.imread('/home/jzhou23/DyCP/f1_data/coffee_f2/train/cam0.png',cv2.IMREAD_UNCHANGED)

#img_dy_result = cv2.imread('/home/jzhou23/DyCP/f1_data/coffee_f1_old/train/cam0.png',cv2.IMREAD_UNCHANGED)

#img_lego_result = cv2.imread('/home/yl4090_1/Tri-MipRF-CP/log2/nerf_synthetic/lego_f20/Tri-MipRF/2024-09-03_15-58-36/rendering/rgb/r_0.png',cv2.IMREAD_UNCHANGED)

#print("----------------------------------")

#print(img_dy,img_dy_result)
#print(img_dy.shape,img_dy_result.shape)

def checkckpt(pth):
    return torch.load(pth)


def resize_by_divider(img, divider):
    h, w = img.shape[:2]
    new_w = w // divider
    new_h = h // divider
    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


import os
import cv2
import shutil


def resize_nerf_scene_by_divider(
    scene_dir,
    output_root,
    divider=3,
    overwrite=False,
):
    """
    Resize PNGs in NeRF synthetic dataset using integer divider.
    Keeps filenames, keeps train/val/test, copies JSON as-is.
    """
    scene_name = os.path.basename(scene_dir.rstrip("/"))
    out_scene = os.path.join(output_root, scene_name)

    os.makedirs(out_scene, exist_ok=True)

    # 1. copy JSON files
    for fname in os.listdir(scene_dir):
        if fname.endswith(".json"):
            src = os.path.join(scene_dir, fname)
            dst = os.path.join(out_scene, fname)
            if not os.path.exists(dst) or overwrite:
                shutil.copy2(src, dst)

    # 2. process train / val / test
    for split in ["train", "val", "test"]:
        in_split = os.path.join(scene_dir, split)
        if not os.path.isdir(in_split):
            continue

        out_split = os.path.join(out_scene, split)
        os.makedirs(out_split, exist_ok=True)

        for fname in os.listdir(in_split):
            if not fname.lower().endswith(".png"):
                continue

            src_img = os.path.join(in_split, fname)
            dst_img = os.path.join(out_split, fname)

            if os.path.exists(dst_img) and not overwrite:
                continue

            img = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)
            if img is None:
                print("[WARN] Cannot read:", src_img)
                continue

            resized = resize_by_divider(img, divider)
            cv2.imwrite(dst_img, resized)

import json

def scale_nerf_intrinsics(json_in, json_out, scale):
    with open(json_in, "r") as f:
        data = json.load(f)

    for k in ["w", "h", "fl_x", "fl_y", "cx", "cy"]:
        if k in data:
            data[k] *= scale

    # enforce integer image size
    data["w"] = int(round(data["w"]))
    data["h"] = int(round(data["h"]))

    with open(json_out, "w") as f:
        json.dump(data, f, indent=2)


#pth_ok= "./test/fd1/nerf_synthetic/coffee_f1/Tri-MipRF/model_w_ds/fm.ckpt"
#pth = "./test/nerf_synthetic/coffee_f1/Tri-MipRF/model_w_ds/fm.ckpt"
#f1 = checkckpt(pth)
#f2 = checkckpt(pth_ok)

#print(f1.shape, f2.shape)
#print(f1)
#print(f2)

resize_nerf_scene_by_divider(
    scene_dir="./f1_data/coffee_f2",
    output_root="./f1_data/coffee_dr",
    divider = 3
)
