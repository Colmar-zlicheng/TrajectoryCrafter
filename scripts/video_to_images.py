import os
import cv2
import argparse
import shutil


def convert_video_to_jpgs(video_path, save_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_id = 0
    # still_reading = True
    still_reading, image = video_capture.read()
    while still_reading:
        img_path = os.path.join(save_path, f'{frame_id:04d}.jpg')
        cv2.imwrite(img_path, image)
        still_reading, image = video_capture.read()
        frame_id += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_path', required=True)
    parser.add_argument('-s', '--save_path', required=True)
    arg = parser.parse_args()

    if os.path.exists(f"{arg.video_path}"):
        video_path = arg.video_path
    elif os.path.exists(f"{arg.video_path}.mp4"):
        video_path = f"{arg.video_path}.mp4"
    elif os.path.exists(f"{arg.video_path}.MOV"):
        video_path = f"{arg.video_path}.MOV"
    else:
        raise ValueError(f'video: {arg.video_path}.xxx do not exist!')

    if os.path.exists(arg.save_path):
        if input(f'Directory {arg.save_path} already exists. Override? [Y/N]: ').lower() == 'y':
            shutil.rmtree(arg.save_path)
        else:
            exit()
    os.makedirs(arg.save_path)
    convert_video_to_jpgs(video_path, arg.save_path)
