import gc
import os
import torch
from models.infer import DepthCrafterDemo
import numpy as np
import torch
import json
from transformers import T5EncoderModel
from omegaconf import OmegaConf
from PIL import Image
from models.crosstransformer3d import CrossTransformer3DModel
from models.autoencoder_magvit import AutoencoderKLCogVideoX
from models.pipeline_trajectorycrafter import TrajCrafter_Pipeline
from models.utils import *
from diffusers import (
    AutoencoderKL,
    CogVideoXDDIMScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration


class TrajCrafter:

    def __init__(self, opts, gradio=False):
        self.funwarp = Warper(device=opts.device)
        # self.depth_estimater = VDADemo(pre_train_path=opts.pre_train_path_vda,device=opts.device)
        self.depth_estimater = DepthCrafterDemo(
            unet_path=opts.unet_path,
            pre_train_path=opts.pre_train_path,
            cpu_offload=opts.cpu_offload,
            device=opts.device,
        )
        self.caption_processor = AutoProcessor.from_pretrained(opts.blip_path)
        self.captioner = Blip2ForConditionalGeneration.from_pretrained(opts.blip_path,
                                                                       torch_dtype=torch.float16).to(opts.device)
        self.setup_diffusion(opts)
        if gradio:
            self.opts = opts

    def infer_custom(self, opts, use_vggt=True):
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)  # (N, 576, 1024, 3)
        prompt = self.get_caption(opts, frames[opts.video_length // 2])

        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
                 )  # [N, 576, 1024, 3] -> [N, 3, 576, 1024], [-1,1]

        if opts.video_length == -1:
            opts.video_length = frames.shape[0]
        else:
            assert frames.shape[0] == opts.video_length

        assert use_vggt

        if use_vggt:
            import roma
            from models.vggt import inference_vggt
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            assert opts.images_path is not None
            assert opts.camera_path is not None
            # TODO:
            ori_h = ori_w = 1280
            resize = False

            vggt_predictions = inference_vggt(opts.images_path)
            vggt_depth = vggt_predictions['depth'].squeeze().detach().cpu().numpy()  # [N, 518, 518]

            vggt_extrinsic, vggt_intrinsic = pose_encoding_to_extri_intri(vggt_predictions['pose_enc'], (576, 1024))
            vggt_intrinsic = vggt_intrinsic.squeeze(0).detach().cpu()  # [N*, 3, 3]

            depths = []
            K = []
            for i in range(vggt_depth.shape[0]):
                intrin = vggt_intrinsic[i].clone()
                if resize:
                    depths.append(
                        torch.from_numpy(cv2.resize(vggt_depth[i], (1024, 576), interpolation=cv2.INTER_NEAREST)))

                    s_x = 1024 / 518
                    s_y = 576 / 518

                    intrin[0, 0] *= s_x  # fx
                    intrin[1, 1] *= s_y  # fy
                    intrin[0, 2] *= s_x  # cx
                    intrin[1, 2] *= s_y  # cy
                else:
                    depths.append(torch.from_numpy(vggt_depth[i]))
                    frames = vggt_predictions['images'].squeeze(0)[:opts.video_length] * 2 - 1

                K.append(intrin)

            depths = torch.stack(depths).to(opts.device).unsqueeze(1)[:opts.video_length]
            K = torch.stack(K).to(opts.device)[:opts.video_length]

            with open(opts.camera_path, 'r') as f:
                cam_data = json.load(f)

            source_cam = torch.tensor(cam_data[opts.images_path.split('/')[-1]], dtype=torch.float32)
            target_cam = torch.tensor(cam_data[opts.target_camera], dtype=torch.float32)

            vggt_extrinsic = vggt_extrinsic.squeeze(0).detach().cpu()[:opts.video_length]  # [N, 3, 4], w2c
            vggt_extrinsic = torch.cat(
                [vggt_extrinsic, torch.tensor([[[0, 0, 0, 1]]]).repeat(opts.video_length, 1, 1)], dim=1)  # [N, 4, 4]
            vggt_extrinsic = torch.linalg.inv(vggt_extrinsic)
            source_cam = source_cam[:opts.video_length]  # [N, 4, 4]
            target_cam = target_cam[:opts.video_length]  # [N, 4, 4]

            # `\sum_i w_i \|s R x_i + t - y_i\|^2`
            # s R x + t = y
            R, t, scale = roma.rigid_points_registration(
                x=source_cam[:, :3, 3],
                y=vggt_extrinsic[:, :3, 3],
                weights=None,
                compute_scaling=True,
            )
            # import open3d as o3d
            # colors = np.zeros((49, 3))
            # normalized_indices = np.arange(49) / (49 - 1)
            # colors[:, 0] = normalized_indices
            # colors[:, 2] = 1 - normalized_indices
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(source_cam[:, :3, 3].numpy())
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "source_cam.ply"), pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(vggt_extrinsic[:, :3, 3].numpy())
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "vggt_extrinsic.ply"), pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(target_cam[:, :3, 3].numpy())
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "target_cam.ply"), pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(
            #     torch.cat([source_cam[:, :3, 3], target_cam[:, :3, 3]], dim=0).numpy())
            # pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, colors], axis=0))
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "source_target_cam.ply"), pcd)

            R_trans = torch.eye(4)
            R_trans[:3, :3] = R
            t_trans = torch.zeros(4)
            t_trans[:3] = t

            # pose_s = scale * (R_trans @ source_cam) + t_trans
            # pose_t = scale * (R_trans @ target_cam) + t_trans

            pose_s = torch.eye(4).unsqueeze(0).repeat(opts.video_length, 1, 1)
            pose_s[:, :3, :3] = R.unsqueeze(0) @ source_cam[:, :3, :3]
            pose_s[:, :3, 3] = (scale * R @ source_cam[:, :3, 3].transpose(0, 1)).transpose(0, 1) + t

            pose_t = torch.eye(4).unsqueeze(0).repeat(opts.video_length, 1, 1)
            pose_t[:, :3, :3] = R.unsqueeze(0) @ target_cam[:, :3, :3]
            pose_t[:, :3, 3] = (scale * R @ target_cam[:, :3, 3].transpose(0, 1)).transpose(0, 1) + t

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pose_s[:, :3, 3].numpy())
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "pose_s.ply"), pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pose_t[:, :3, 3].numpy())
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "pose_t.ply"), pcd)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(torch.cat([pose_s[:, :3, 3], pose_t[:, :3, 3]], dim=0).numpy())
            # pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, colors], axis=0))
            # o3d.io.write_point_cloud(os.path.join(opts.save_dir, "pose_s_t.ply"), pcd)

            pose_s = torch.linalg.inv(pose_s)
            pose_t = torch.linalg.inv(pose_t)

            # error = torch.mean(torch.linalg.norm(pose_s[:, :3, 3] - vggt_extrinsic[:, :3, 3], axis=1))
            # print(torch.linalg.inv(pose_s) @ vggt_extrinsic)
            # print(error)
            # print(torch.mean(torch.linalg.norm(vggt_extrinsic[:, :3, 3], axis=1)))
            # print(torch.mean(torch.linalg.norm(pose_s[:, :3, 3], axis=1)))
            # exit()
        else:
            depths = self.depth_estimater.infer(
                frames,
                opts.near,
                opts.far,
                opts.depth_inference_steps,
                opts.depth_guidance_scale,
                window_size=opts.window_size,
                overlap=opts.overlap,
            ).to(opts.device)  # [N, 1, 576, 1024]

            pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
            # print(pose_s.shape) # [N，4, 4]
            # print(pose_t.shape) # [N 4, 4]
            # print(K.shape) # [N，3, 3]

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i:i + 1],
                None,
                depths[i:i + 1],
                pose_s[i:i + 1],
                pose_t[i:i + 1],
                K[i:i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )
        resized_batch = F.interpolate(sample[0].permute(1, 0, 2, 3),
                                      size=(ori_h, ori_w),
                                      mode='bilinear',
                                      align_corners=False)
        save_video(
            resized_batch.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'gen_rs.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, opts.video_length, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_gradual(self, opts):
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)  # (N, 576, 1024, 3)
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)  # [N, 1, 576, 1024]

        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
                 )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)
        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i:i + 1],
                None,
                depths[i:i + 1],
                pose_s[i:i + 1],
                pose_t[i:i + 1],
                K[i:i + 1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, 49, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_direct(self, opts):
        opts.cut = 20
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
                 )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.cut)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            if i < opts.cut:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[0:1],
                    None,
                    depths[0:1],
                    pose_s[0:1],
                    pose_t[i:i + 1],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
            else:
                warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                    frames[i - opts.cut:i - opts.cut + 1],
                    None,
                    depths[i - opts.cut:i - opts.cut + 1],
                    pose_s[0:1],
                    pose_t[-1:],
                    K[0:1],
                    None,
                    opts.mask,
                    twice=False,
                )
                warped_images.append(warped_frame2)
                masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames[:opts.video_length - opts.cut].permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video[opts.cut:].permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks[opts.cut:].repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0)[opts.cut:],
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0][:, :opts.video_length - opts.cut, ...].to(opts.device)
            tensor_right = sample[0][:, opts.cut:, ...].to(opts.device)
            interval = torch.ones(3, opts.video_length - opts.cut, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def infer_bullet(self, opts):
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)

        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
                 )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses(opts, depths, num_frames=opts.video_length)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[-1:],
                None,
                depths[-1:],
                pose_s[0:1],
                pose_t[i:i + 1],
                K[0:1],
                None,
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)
        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )
        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, -10:, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_left_full = torch.cat([tensor_left, tensor_left[:, -1:, :, :].repeat(1, 48, 1, 1)], dim=1)
            tensor_right = sample[0].to(opts.device)
            tensor_right_full = torch.cat([tensor_left, tensor_right[:, 1:, :, :]], dim=1)
            interval = torch.ones(3, 49 * 2 - 1, 384, 30).to(opts.device)
            result = torch.cat((tensor_left_full, interval, tensor_right_full), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 4,
            )

    def infer_zoom(self, opts):
        frames = read_video_frames(opts.video_path, opts.video_length, opts.stride, opts.max_res)
        prompt = self.get_caption(opts, frames[opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            opts.near,
            opts.far,
            opts.depth_inference_steps,
            opts.depth_guidance_scale,
            window_size=opts.window_size,
            overlap=opts.overlap,
        ).to(opts.device)
        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(opts.device) * 2.0 - 1.0
                 )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        assert frames.shape[0] == opts.video_length
        pose_s, pose_t, K = self.get_poses_f(opts, depths, num_frames=opts.video_length, f_new=250)

        warped_images = []
        masks = []
        for i in tqdm(range(opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i:i + 1],
                None,
                depths[i:i + 1],
                pose_s[i:i + 1],
                pose_t[i:i + 1],
                K[0:1],
                K[i:i + 1],
                opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(frames, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(opts.save_dir, 'input.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'render.mp4'),
            fps=opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(opts.save_dir, 'mask.mp4'),
            fps=opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=opts.device).manual_seed(opts.seed)

        del self.depth_estimater
        del self.caption_processor
        del self.captioner
        gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=opts.video_length,
                negative_prompt=opts.negative_prompt,
                height=opts.sample_size[0],
                width=opts.sample_size[1],
                generator=generator,
                guidance_scale=opts.diffusion_guidance_scale,
                num_inference_steps=opts.diffusion_inference_steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(opts.save_dir, 'gen.mp4'),
            fps=opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(opts.device)
            tensor_right = sample[0].to(opts.device)
            interval = torch.ones(3, 49, 384, 30).to(opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(opts.save_dir, 'viz.mp4'),
                fps=opts.fps * 2,
            )

    def get_caption(self, opts, image):
        image_array = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_array)
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(opts.device, torch.float16)
        generated_ids = self.captioner.generate(**inputs)
        generated_text = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text + opts.refine_prompt

    def get_poses(self, opts, depths, num_frames):
        radius = (depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu() * opts.radius_scale)
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]).repeat(num_frames, 1, 1).to(opts.device))
        c2w_init = (torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]).to(opts.device).unsqueeze(0))
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device)
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx:opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def get_poses_f(self, opts, depths, num_frames, f_new):
        radius = (depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu() * opts.radius_scale)
        radius = min(radius, 5)
        cx = 512.0
        cy = 288.0
        f = 500
        # f_new,d_r: 250,0.5; 1000,-0.9
        f_values = torch.linspace(f, f_new, num_frames, device=opts.device)
        K = torch.zeros((num_frames, 3, 3), device=opts.device)
        K[:, 0, 0] = f_values
        K[:, 1, 1] = f_values
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        c2w_init = (torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]).to(opts.device).unsqueeze(0))
        if opts.camera == 'target':
            dtheta, dphi, dr, dx, dy = opts.target_pose
            poses = generate_traj_specified(c2w_init, dtheta, dphi, dr * radius, dx, dy, num_frames, opts.device)
        elif opts.camera == 'traj':
            with open(opts.traj_txt, 'r') as file:
                lines = file.readlines()
                theta = [float(i) for i in lines[0].split()]
                phi = [float(i) for i in lines[1].split()]
                r = [float(i) * radius for i in lines[2].split()]
            poses = generate_traj_txt(c2w_init, phi, theta, r, num_frames, opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[opts.anchor_idx:opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses
        return pose_s, pose_t, K

    def setup_diffusion(self, opts):
        # transformer = CrossTransformer3DModel.from_pretrained_cus(opts.transformer_path).to(opts.weight_dtype)
        transformer = CrossTransformer3DModel.from_pretrained(opts.transformer_path).to(opts.weight_dtype)
        # transformer = transformer.to(opts.weight_dtype)
        vae = AutoencoderKLCogVideoX.from_pretrained(opts.model_name, subfolder="vae").to(opts.weight_dtype)
        text_encoder = T5EncoderModel.from_pretrained(opts.model_name,
                                                      subfolder="text_encoder",
                                                      torch_dtype=opts.weight_dtype)
        # Get Scheduler
        Choosen_Scheduler = {
            "Euler": EulerDiscreteScheduler,
            "Euler A": EulerAncestralDiscreteScheduler,
            "DPM++": DPMSolverMultistepScheduler,
            "PNDM": PNDMScheduler,
            "DDIM_Cog": CogVideoXDDIMScheduler,
            "DDIM_Origin": DDIMScheduler,
        }[opts.sampler_name]
        scheduler = Choosen_Scheduler.from_pretrained(opts.model_name, subfolder="scheduler")

        self.pipeline = TrajCrafter_Pipeline.from_pretrained(
            opts.model_name,
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=opts.weight_dtype,
        )

        if opts.low_gpu_memory_mode:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.enable_model_cpu_offload()

    def run_gradio(self, input_video, stride, radius_scale, pose, steps, seed):
        frames = read_video_frames(input_video, self.opts.video_length, stride, self.opts.max_res)
        prompt = self.get_caption(self.opts, frames[self.opts.video_length // 2])
        # depths= self.depth_estimater.infer(frames, opts.near, opts.far).to(opts.device)
        depths = self.depth_estimater.infer(
            frames,
            self.opts.near,
            self.opts.far,
            self.opts.depth_inference_steps,
            self.opts.depth_guidance_scale,
            window_size=self.opts.window_size,
            overlap=self.opts.overlap,
        ).to(self.opts.device)
        frames = (torch.from_numpy(frames).permute(0, 3, 1, 2).to(self.opts.device) * 2.0 - 1.0
                 )  # 49 576 1024 3 -> 49 3 576 1024, [-1,1]
        num_frames = frames.shape[0]
        assert num_frames == self.opts.video_length
        radius_scale = float(radius_scale)
        radius = (depths[0, 0, depths.shape[-2] // 2, depths.shape[-1] // 2].cpu() * radius_scale)
        radius = min(radius, 5)
        cx = 512.0  # depths.shape[-1]//2
        cy = 288.0  # depths.shape[-2]//2
        f = 500  # 500.
        K = (torch.tensor([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]).repeat(num_frames, 1, 1).to(self.opts.device))
        c2w_init = (torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]).to(self.opts.device).unsqueeze(0))

        # import pdb
        # pdb.set_trace()
        theta, phi, r, x, y = [float(i) for i in pose.split(';')]
        # theta,phi,r,x,y = [float(i) for i in theta.split()],[float(i)
        # for i in phi.split()],[float(i)
        # for i in r.split()],[float(i) for i in x.split()],[float(i) for i in y.split()]
        # target mode
        poses = generate_traj_specified(c2w_init, theta, phi, r * radius, x, y, num_frames, self.opts.device)
        poses[:, 2, 3] = poses[:, 2, 3] + radius
        pose_s = poses[self.opts.anchor_idx:self.opts.anchor_idx + 1].repeat(num_frames, 1, 1)
        pose_t = poses

        warped_images = []
        masks = []
        for i in tqdm(range(self.opts.video_length)):
            warped_frame2, mask2, warped_depth2, flow12 = self.funwarp.forward_warp(
                frames[i:i + 1],
                None,
                depths[i:i + 1],
                pose_s[i:i + 1],
                pose_t[i:i + 1],
                K[i:i + 1],
                None,
                self.opts.mask,
                twice=False,
            )
            warped_images.append(warped_frame2)
            masks.append(mask2)
        cond_video = (torch.cat(warped_images) + 1.0) / 2.0
        cond_masks = torch.cat(masks)

        frames = F.interpolate(frames, size=self.opts.sample_size, mode='bilinear', align_corners=False)
        cond_video = F.interpolate(cond_video, size=self.opts.sample_size, mode='bilinear', align_corners=False)
        cond_masks = F.interpolate(cond_masks, size=self.opts.sample_size, mode='nearest')
        save_video(
            (frames.permute(0, 2, 3, 1) + 1.0) / 2.0,
            os.path.join(self.opts.save_dir, 'input.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_video.permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'render.mp4'),
            fps=self.opts.fps,
        )
        save_video(
            cond_masks.repeat(1, 3, 1, 1).permute(0, 2, 3, 1),
            os.path.join(self.opts.save_dir, 'mask.mp4'),
            fps=self.opts.fps,
        )

        frames = (frames.permute(1, 0, 2, 3).unsqueeze(0) + 1.0) / 2.0
        frames_ref = frames[:, :, :10, :, :]
        cond_video = cond_video.permute(1, 0, 2, 3).unsqueeze(0)
        cond_masks = (1.0 - cond_masks.permute(1, 0, 2, 3).unsqueeze(0)) * 255.0
        generator = torch.Generator(device=self.opts.device).manual_seed(seed)

        # del self.depth_estimater
        # del self.caption_processor
        # del self.captioner
        # gc.collect()
        torch.cuda.empty_cache()
        with torch.no_grad():
            sample = self.pipeline(
                prompt,
                num_frames=self.opts.video_length,
                negative_prompt=self.opts.negative_prompt,
                height=self.opts.sample_size[0],
                width=self.opts.sample_size[1],
                generator=generator,
                guidance_scale=self.opts.diffusion_guidance_scale,
                num_inference_steps=steps,
                video=cond_video,
                mask_video=cond_masks,
                reference=frames_ref,
            ).videos
        save_video(
            sample[0].permute(1, 2, 3, 0),
            os.path.join(self.opts.save_dir, 'gen.mp4'),
            fps=self.opts.fps,
        )

        viz = True
        if viz:
            tensor_left = frames[0].to(self.opts.device)
            tensor_right = sample[0].to(self.opts.device)
            interval = torch.ones(3, 49, 384, 30).to(self.opts.device)
            result = torch.cat((tensor_left, interval, tensor_right), dim=3)
            result_reverse = torch.flip(result, dims=[1])
            final_result = torch.cat((result, result_reverse[:, 1:, :, :]), dim=1)
            save_video(
                final_result.permute(1, 2, 3, 0),
                os.path.join(self.opts.save_dir, 'viz.mp4'),
                fps=self.opts.fps * 2,
            )
        return os.path.join(self.opts.save_dir, 'viz.mp4')
