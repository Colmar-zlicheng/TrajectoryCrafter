import os
import torch
import sys
from demo import TrajCrafter
import random
import gradio as gr
import random
from inference import get_parser
from datetime import datetime
import argparse

# 解析命令行参数

traj_examples = [
    ['20; -30; 0.3; 0; 0'],
    ['0; 0; -0.3; -2; 2'],
]

# inputs=[i2v_input_video, i2v_stride, i2v_center_scale, i2v_pose, i2v_steps, i2v_seed],

img_examples = [
    ['test/videos/0-NNvgaTcVzAG0-r.mp4', 2, 1, '0; -30; 0.5; -2; 0', 50, 43],
    ['test/videos/tUfDESZsQFhdDW9S.mp4', 2, 1, '0; 30; -0.4; 2; 0', 50, 43],
    ['test/videos/part-2-3.mp4', 2, 1, '20; 40; 0.5; 2; 0', 50, 43],
    ['test/videos/p7.mp4', 2, 1, '0; -50; 0.3; 0; 0', 50, 43],
    ['test/videos/UST-fn-RvhJwMR5S.mp4', 2, 1, '0; -35; 0.4; 0; 0', 50, 43],
]

max_seed = 2**31

parser = get_parser()  # infer_config.py
opts = parser.parse_args()  # default device: 'cuda:0'
opts.weight_dtype = torch.bfloat16
tmp = datetime.now().strftime("%Y%m%d_%H%M")
opts.save_dir = f'./experiments/gradio_{tmp}'
os.makedirs(opts.save_dir, exist_ok=True)
test_tensor = torch.Tensor([0]).cuda()
opts.device = str(test_tensor.device)

CAMERA_MOTION_MODE = ["Basic Camera Trajectory", "Custom Camera Trajectory"]


def show_traj(mode):
    if mode == 'Orbit Left':
        return gr.update(value='0; -30; 0; 0; 0', visible=True), gr.update(visible=False)
    elif mode == 'Orbit Right':
        return gr.update(value='0; 30; 0; 0; 0', visible=True), gr.update(visible=False)
    elif mode == 'Orbit Up':
        return gr.update(value='30; 0; 0; 0; 0', visible=True), gr.update(visible=False)
    elif mode == 'Orbit Down':
        return gr.update(value='-20; 0; 0; 0; 0', visible=True), gr.update(visible=False)
    if mode == 'Pan Left':
        return gr.update(value='0; 0; 0; -2; 0', visible=True), gr.update(visible=False)
    elif mode == 'Pan Right':
        return gr.update(value='0; 0; 0; 2; 0', visible=True), gr.update(visible=False)
    elif mode == 'Pan Up':
        return gr.update(value='0; 0; 0; 0; 2', visible=True), gr.update(visible=False)
    elif mode == 'Pan Down':
        return gr.update(value='0; 0; 0; 0; -2', visible=True), gr.update(visible=False)
    elif mode == 'Zoom in':
        return gr.update(value='0; 0; 0.5; 0; 0', visible=True), gr.update(visible=False)
    elif mode == 'Zoom out':
        return gr.update(value='0; 0; -0.5; 0; 0', visible=True), gr.update(visible=False)
    elif mode == 'Customize':
        return gr.update(value='0; 0; 0; 0; 0', visible=True), gr.update(visible=True)
    elif mode == 'Reset':
        return gr.update(value='0; 0; 0; 0; 0', visible=False), gr.update(visible=False)


def trajcrafter_demo(opts):
    css = """
    #input_img {max-width: 1024px !important} 
    #output_vid {max-width: 1024px; max-height:576px} 
    #random_button {max-width: 100px !important}
    .generate-btn {
        background: linear-gradient(45deg, #2196F3, #1976D2) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
    }
    .generate-btn:hover {
        background: linear-gradient(45deg, #1976D2, #1565C0) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    """
    image2video = TrajCrafter(opts, gradio=True)
    # image2video.run_both = spaces.GPU(image2video.run_both, duration=290) # fixme
    with gr.Blocks(analytics_enabled=False, css=css) as trajcrafter_iface:
        gr.Markdown("""
            <div align='center'>
                <h1>TrajectoryCrafter: Redirecting View Trajectory for Monocular Videos via Diffusion Models</h1>
                <a style='font-size:18px;color: #FF5DB0' href='https://github.com/TrajectoryCrafter/TrajectoryCrafter'>[Github]</a>
                <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2409.02048'>[ArXiv]</a>
                <a style='font-size:18px;color: #000000' href='https://trajectorycrafter.github.io/'>[Project Page]</a>
                <a style='font-size:18px;color: #000000' href='https://www.youtube.com/watch?v=dQtHFgyrids'>[Video]</a>
            </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column():
                # # step 1: input an image
                # gr.Markdown("---\n## Step 1: Input an Image, selet an elevation angle and a center_scale factor", show_label=False, visible=True)
                # gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>1. Estimate an elevation angle  that represents the angle at which the image was taken; a value bigger than 0 indicates a top-down view, and it doesn't need to be precise. <br>2. The origin of the world coordinate system is by default defined at the point cloud corresponding to the center pixel of the input image. You can adjust the position of the origin by modifying center_scale; a value smaller than 1 brings the origin closer to you.</div>")
                i2v_input_video = gr.Video(label="Input Video", elem_id="input_video", format="mp4")

            with gr.Column():
                i2v_output_video = gr.Video(
                    label="Generated Video",
                    elem_id="output_vid",
                    autoplay=True,
                    show_share_button=True,
                )

        with gr.Row():
            with gr.Row():
                i2v_stride = gr.Slider(
                    minimum=1,
                    maximum=3,
                    step=1,
                    elem_id="stride",
                    label="Stride",
                    value=1,
                )
                i2v_center_scale = gr.Slider(
                    minimum=0.1,
                    maximum=2,
                    step=0.1,
                    elem_id="i2v_center_scale",
                    label="center_scale",
                    value=1,
                )
                i2v_steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    elem_id="i2v_steps",
                    label="Sampling steps",
                    value=50,
                )
                i2v_seed = gr.Slider(label='Random seed', minimum=0, maximum=max_seed, step=1, value=43)
            with gr.Row():
                pan_left = gr.Button(value="Pan Left")
                pan_right = gr.Button(value="Pan Right")
                pan_up = gr.Button(value="Pan Up")
                pan_down = gr.Button(value="Pan Down")
            with gr.Row():
                orbit_left = gr.Button(value="Orbit Left")
                orbit_right = gr.Button(value="Orbit Right")
                orbit_up = gr.Button(value="Orbit Up")
                orbit_down = gr.Button(value="Orbit Down")
            with gr.Row():
                zin = gr.Button(value="Zoom in")
                zout = gr.Button(value="Zoom out")
                custom = gr.Button(value="Customize")
                reset = gr.Button(value="Reset")
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        i2v_pose = gr.Text(
                            value='0; 0; 0; 0; 0',
                            label="Traget camera pose (theta, phi, r, x, y)",
                            visible=False,
                        )
                        with gr.Column(visible=False) as i2v_egs:
                            gr.Markdown(
                                "<div align='left' style='font-size:18px;color: #000000'>Please refer to <a href='https://github.com/TrajectoryCrafter/TrajectoryCrafter/blob/main/docs/config_help.md' target='_blank'>tutorial</a> for customizing camera trajectory.</div>"
                            )
                            gr.Examples(
                                examples=traj_examples,
                                inputs=[i2v_pose],
                            )
            with gr.Column():
                i2v_end_btn = gr.Button(
                    "Generate video",
                    scale=2,
                    size="lg",
                    variant="primary",
                    elem_classes="generate-btn",
                )

                # with gr.Column():
                #     i2v_input_video = gr.Video(label="Input Video", elem_id="input_video", format="mp4")
                # i2v_input_image = gr.Image(label="Input Image",elem_id="input_img")
                # with gr.Row():
                #     # i2v_elevation = gr.Slider(minimum=-45, maximum=45, step=1, elem_id="elevation", label="elevation", value=5)
                #     i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="center_scale", value=1)
                #     i2v_steps = gr.Slider(minimum=1, maximum=50, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                #     i2v_seed = gr.Slider(label='Random seed', minimum=0, maximum=max_seed, step=1, value=43)
                # with  gr.Column():
                #     with gr.Row():
                #         left = gr.Button(value = "Left")
                #         right = gr.Button(value = "Right")
                #         up = gr.Button(value = "Up")
                #     with gr.Row():
                #         down = gr.Button(value = "Down")
                #         zin = gr.Button(value = "Zoom in")
                #         zout = gr.Button(value = "Zoom out")
                #     with gr.Row():
                #         custom = gr.Button(value = "Customize")
                #         reset = gr.Button(value = "Reset")

            # step 3 - Generate video
            # with gr.Column():
            # gr.Markdown("---\n## Step 3: Generate video", show_label=False, visible=True)
            # gr.Markdown("<div align='left' style='font-size:18px;color: #000000'> You can reduce the sampling steps for faster inference; try different random seed if the result is not satisfying. </div>")
            # i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)
            # i2v_end_btn = gr.Button("Generate video")
            # i2v_traj_video = gr.Video(label="Camera Trajectory",elem_id="traj_vid",autoplay=True,show_share_button=True)

            # with  gr.Column(scale=1.5):
            # with gr.Row():
            #     # i2v_elevation = gr.Slider(minimum=-45, maximum=45, step=1, elem_id="elevation", label="elevation", value=5)
            #     i2v_center_scale = gr.Slider(minimum=0.1, maximum=2, step=0.1, elem_id="i2v_center_scale", label="center_scale", value=1)
            #     i2v_steps = gr.Slider(minimum=1, maximum=50, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
            #     i2v_seed = gr.Slider(label='Random seed', minimum=0, maximum=max_seed, step=1, value=43)
            # with gr.Row():
            #     pan_left = gr.Button(value = "Pan Left")
            #     pan_right = gr.Button(value = "Pan Right")
            #     pan_up = gr.Button(value = "Pan Up")
            #     pan_down = gr.Button(value = "Pan Down")
            # with gr.Row():
            #     orbit_left = gr.Button(value = "Orbit Left")
            #     orbit_right = gr.Button(value = "Orbit Right")
            #     orbit_up = gr.Button(value = "Orbit Up")
            #     orbit_down = gr.Button(value = "Orbit Down")
            # with gr.Row():
            #     zin = gr.Button(value = "Zoom in")
            #     zout = gr.Button(value = "Zoom out")
            #     custom = gr.Button(value = "Customize")
            #     reset = gr.Button(value = "Reset")
            # with gr.Column():
            #      with gr.Row():
            #         with gr.Column():
            #             i2v_pose = gr.Text(value = '0; 0; 0; 0; 0', label="Traget camera pose (theta, phi, r, x, y)",visible=False)
            #             with gr.Column(visible=False) as i2v_egs:
            #                 gr.Markdown("<div align='left' style='font-size:18px;color: #000000'>Please refer to the <a href='https://github.com/Drexubery/ViewCrafter/blob/main/docs/gradio_tutorial.md' target='_blank'>tutorial</a> for customizing camera trajectory.</div>")
            #                 gr.Examples(examples=traj_examples,
            #                         inputs=[i2v_pose],
            #                     )
            # with gr.Row():
            #     i2v_end_btn = gr.Button("Generate video")
        # step 3 - Generate video
        # with gr.Row():
        #     with gr.Column():

        i2v_end_btn.click(
            inputs=[
                i2v_input_video,
                i2v_stride,
                i2v_center_scale,
                i2v_pose,
                i2v_steps,
                i2v_seed,
            ],
            outputs=[i2v_output_video],
            fn=image2video.run_gradio,
        )

        pan_left.click(inputs=[pan_left], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        pan_right.click(inputs=[pan_right], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        pan_up.click(inputs=[pan_up], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        pan_down.click(inputs=[pan_down], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        orbit_left.click(inputs=[orbit_left], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        orbit_right.click(inputs=[orbit_right], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        orbit_up.click(inputs=[orbit_up], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        orbit_down.click(inputs=[orbit_down], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        zin.click(inputs=[zin], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        zout.click(inputs=[zout], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        custom.click(inputs=[custom], outputs=[i2v_pose, i2v_egs], fn=show_traj)
        reset.click(inputs=[reset], outputs=[i2v_pose, i2v_egs], fn=show_traj)

        gr.Examples(
            examples=img_examples,
            # inputs=[i2v_input_video,i2v_stride],
            inputs=[
                i2v_input_video,
                i2v_stride,
                i2v_center_scale,
                i2v_pose,
                i2v_steps,
                i2v_seed,
            ],
        )

    return trajcrafter_iface


trajcrafter_iface = trajcrafter_demo(opts)
trajcrafter_iface.queue(max_size=10)
# trajcrafter_iface.launch(server_name=args.server_name, max_threads=10, debug=True)
trajcrafter_iface.launch(server_name="0.0.0.0", server_port=12345, debug=True, share=False, max_threads=10)
