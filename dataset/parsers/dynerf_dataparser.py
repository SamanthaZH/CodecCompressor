from pathlib import Path
import numpy as np
import imageio
import dataset.utils.io as data_io
from dataset.utils.cameras import PinholeCamera


def load_data(base_path: Path, scene: str, split: str):
    data_path = base_path / scene
    splits = ['train', 'val'] if split == "trainval" else [split]
    meta = None
    for s in splits:
        meta_path = data_path / "transforms_{}.json".format(s)
        m = data_io.load_from_json(meta_path)
        if meta is None:
            meta = m
        else:
            for k, v in meta.items():
                if type(v) is list:
                    v.extend(m[k])
                else:
                    assert v == m[k]

    image_height = meta['h']
    image_width = meta['w']
    #image_height, image_width = 800, 800
    if "camera_angle_x" in meta:
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length_x = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        focal_length_y = focal_length_x
    else:
        focal_length_x = meta['fl_x']
        focal_length_y = meta['fl_y']
    if 'cx' not in meta:
        cx = image_width / 2.0
        cy = image_height / 2.0

    else:
        cx = meta['cx']
        cy = meta['cy']
    cameras = [
        PinholeCamera(
            fx=focal_length_x,
            fy=focal_length_y,
            cx=cx,
            cy=cy,
            width=int(image_width),
            height=int(image_height),
        )
    ]
    cam_num = len(cameras)

    frames, poses = {k: [] for k in range(len(cameras))}, {
        k: [] for k in range(len(cameras))
    }
    index = 0
    for frame in meta["frames"]:
        fname = data_path / Path(frame["file_path"].replace("./", "") + ".png")
        frames[index % cam_num].append(
            {
                'image_filename': fname,
                'lossmult': 1.0,
            }
        )
        poses[index % cam_num].append(
            np.array(frame["transform_matrix"]).astype(np.float32)
        )
        index = index + 1

    aabb = np.array([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

    outputs = {
        'frames': frames,
        'poses': poses,
        'cameras': cameras,
        'aabb': aabb,
    }
    return outputs
