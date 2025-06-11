import os
import math
import argparse
import numpy as np
from tqdm import tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_cam', type=str, required=True)
    parser.add_argument('-o', '--output_cam', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_cam, exist_ok=True)

    cam_files = sorted(os.listdir(args.input_cam))

    shift = np.array([
        [1, 0, 0, 0],
        [0, math.sqrt(3) / 2, 0.5, -0.5],
        [0, -0.5, math.sqrt(3) / 2, 0],
        [0, 0, 0, 1],
    ],
                     dtype=np.float32)

    for i, cam_file in enumerate(tqdm(cam_files)):
        camera = np.load(os.path.join(args.input_cam, cam_file))

        intrins = camera['intrinsics']
        c2w = camera['pose']

        c2w_shift = shift @ c2w

        np.savez(
            os.path.join(args.output_cam, cam_file),
            pose=c2w_shift,
            intrinsics=intrins,
        )
