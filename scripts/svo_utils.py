import os
import json
import glob
import torch
import numpy as np
import torchvision
from scipy.spatial.transform import Rotation
from PIL import Image
import h5py
import pickle
import argparse
import pyzed.sl as sl
from tqdm import tqdm


def save_video(data, images_path, folder=None, fps=8):
    if isinstance(data, np.ndarray):
        tensor_data = (torch.from_numpy(data) * 255).to(torch.uint8)
    elif isinstance(data, torch.Tensor):
        tensor_data = (data.detach().cpu() * 255).to(torch.uint8)
    elif isinstance(data, list):
        folder = [folder] * len(data)
        images = [np.array(Image.open(os.path.join(folder_name, path))) for folder_name, path in zip(folder, data)]
        stacked_images = np.stack(images, axis=0)
        tensor_data = torch.from_numpy(stacked_images).to(torch.uint8)
    torchvision.io.write_video(images_path, tensor_data, fps=fps, video_codec='h264', options={'crf': '10'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data/test_data/Fri_Jul__7_09:42:23_2023")
    parser.add_argument('-c', '--camera_id', type=str, required=True)
    args = parser.parse_args()

    extract_dir = os.path.join(args.data_path, "extract")
    extract_cam_dir = os.path.join(args.data_path, "extract/camera")
    extract_depth_dir = os.path.join(args.data_path, "extract/depth")
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(extract_cam_dir, exist_ok=True)
    os.makedirs(extract_depth_dir, exist_ok=True)

    json_file_paths = glob.glob(str(args.data_path) + "/*.json")
    if len(json_file_paths) < 1:
        raise Exception(f"Unable to find metadata file at '{args.data_path}'")
    with open(os.path.join(json_file_paths[0]), "r") as metadata_file:
        metadata = json.load(metadata_file)
    trajectory_length = metadata["trajectory_length"]

    trajectory = h5py.File(os.path.join(args.data_path, 'trajectory.h5'), "r")
    exts = trajectory['observation']['camera_extrinsics']

    camera_id = args.camera_id

    camera_extr_6d = exts[f'{camera_id}_left'][:]
    camera_extr = []
    for i in tqdm(range(camera_extr_6d.shape[0])):
        extr_6d = camera_extr_6d[i]
        rot = Rotation.from_euler("xyz", np.array(extr_6d[3:])).as_matrix()
        trans = np.array(extr_6d[:3])
        additional_row = np.array([0, 0, 0, 1]).reshape(1, 4)
        extrinsics = np.concatenate([rot, trans.reshape(3, 1)], axis=1)
        c2w = np.concatenate([extrinsics, additional_row], axis=0)  # camera-to-world
        camera_extr.append(c2w.tolist())

    with open(os.path.join(extract_cam_dir, f"{camera_id}_extr_left.json"), 'w') as f:
        json.dump(camera_extr, f, indent=4)

    svo_path = os.path.join(args.data_path, "recordings/SVO", f"{camera_id}.svo")
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_path))
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.2

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise Exception(f"Error reading camera data: {err}")

    params = (zed.get_camera_information().camera_configuration.calibration_parameters)

    left_intrinsic_mat = np.array([
        [params.left_cam.fx, 0, params.left_cam.cx],
        [0, params.left_cam.fy, params.left_cam.cy],
        [0, 0, 1],
    ])
    right_intrinsic_mat = np.array([
        [params.right_cam.fx, 0, params.right_cam.cx],
        [0, params.right_cam.fy, params.right_cam.cy],
        [0, 0, 1],
    ])

    with open(os.path.join(extract_cam_dir, f"{camera_id}_intr.json"), 'w') as f:
        json.dump({"left": left_intrinsic_mat.tolist(), "right": right_intrinsic_mat.tolist()}, f, indent=4)

    depths = []

    for i in tqdm(range(trajectory_length)):
        depth_image = sl.Mat()
        rt_param = sl.RuntimeParameters()
        err = zed.grab(rt_param)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)
            depth_image = np.array(depth_image.numpy())
            depth_image = np.nan_to_num(depth_image, nan=0, posinf=1.8, neginf=0)
            depth_image[depth_image > 1.8] = 0

            depths.append(depth_image)
        else:
            depth_image = None
            if i < trajectory_length - 1:
                assert False

    depths = np.stack(depths, axis=0)
    with open(os.path.join(extract_depth_dir, f"{camera_id}_left.pkl"), 'wb') as f:
        pickle.dump(depths, f)
    # print(depths.shape)
    # save_video(
    #     torch.tensor(depths).unsqueeze(-1).repeat(1, 1, 1, 3),
    #     'tmp/b.mp4',
    # )
