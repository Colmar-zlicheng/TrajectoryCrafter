import os
import json
import argparse
import numpy as np


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_path', required=True)
    parser.add_argument('-s', '--save_path', required=True)
    arg = parser.parse_args()

    with open(arg.json_path, 'r') as file:
        cam_data = json.load(file)

    transform_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    cam_list = {}
    for cam_id in range(1, 11):
        tmp_list = []
        for frame_id in range(0, 81):
            cam_str = cam_data[f"frame{frame_id}"][f"cam{cam_id:02}"]

            c2w = parse_matrix(cam_str)

            c2w = c2w.transpose(1, 0)

            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 100

            # c2w = transform_matrix @ c2w
            # c2w = np.linalg.inv(c2w)

            tmp_list.append(c2w.tolist())
        cam_list[f"cam{cam_id:02}"] = tmp_list

    with open(arg.save_path, 'w') as f:
        json.dump(cam_list, f, indent=4)
