# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --video_path './data/test_data/scene15/videos/cam03.mp4' \
#     --images_path './data/test_data/scene15/images/cam03' \
#     --camera_path './data/test_data/scene15/cameras/cameras.json' \
#     --target_camera 'cam01' \
#     --stride 1 \
#     --video_length -1 \
#     --out_dir experiments \
#     --mask \
#     --mode 'custom' \

CUDA_VISIBLE_DEVICES=2 python inference.py \
    --droid_path './data/test_data/Fri_Jul__7_09:42:23_2023' \
    --driod_camera '{"static": [22008760, 24400334], "wrist": 18026681}'\
    --stride 1 \
    --video_length -1 \
    --out_dir experiments \
    --mask \
    --mode 'droid' \

# # gradual mode
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'gradual' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \

# # direct mode
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'direct' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \

# # bullet time
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'bullet' \
#     --mask \
#     --target_pose 0 -30 0.3 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \

# # dolly-zoom mode
# python inference.py \
#     --video_path './test/videos/p7.mp4' \
#     --stride 2 \
#     --out_dir experiments \
#     --radius_scale 1 \
#     --camera 'target' \
#     --mode 'zoom' \
#     --mask \
#     --target_pose 0 0 0.5 0 0 \
#     --traj_txt 'test/trajs/loop2.txt' \