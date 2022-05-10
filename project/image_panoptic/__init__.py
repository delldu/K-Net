"""Image Weather Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch
import numpy as np

import redos
import todos

from . import segment
from . import ade20k

import pdb

SEGMENT_MEAN = [0.485, 0.456, 0.406]
SEGMENT_STD = [0.229, 0.224, 0.225]
SEGMENT_TIMES = 4


def get_model(checkpoint):
    """Create model."""

    model = segment.SegmentModel()
    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    return model, device


# import torchvision.transforms as T
# from PIL import Image
# from einops import rearrange

# def load_gray_tensor(image_file):
#     image = Image.open(image_file).convert("L").convert("RGB")
#     tensor = T.ToTensor()(image)
#     return rearrange(tensor, "C H W -> 1 C H W")


def blender_segment(input_tensor, output_tensor):
    palette = np.array(ade20k.ADE20K.PALETTE)

    # input_tensor.size() -- [3, 512, 512]
    color_numpy = np.zeros((input_tensor.size(1), input_tensor.size(2), 3), dtype=np.uint8)
    mask_numpy = output_tensor.squeeze(0).squeeze(0).numpy()
    for label, color in enumerate(palette):
        color_numpy[mask_numpy == label, :] = color
    color_tensor = torch.from_numpy(color_numpy).permute(2, 0, 1)

    return 0.5 * input_tensor.cpu() + 0.5 * color_tensor / 255.0


def model_forward(model, device, input_tensor):
    # normal_tensor only support CxHxW !!!
    input_tensor = input_tensor.squeeze(0)

    # image_tensor = input_tensor
    image_tensor = torch.zeros_like(input_tensor)
    image_tensor.copy_(input_tensor)

    todos.data.normal_tensor(input_tensor, mean=SEGMENT_MEAN, std=SEGMENT_STD)
    input_tensor = input_tensor.unsqueeze(0)

    # zeropad for model
    H, W = input_tensor.size(2), input_tensor.size(3)
    if H % SEGMENT_TIMES == 0 and W % SEGMENT_TIMES == 0:
        output_tensor = todos.model.forward(model, device, input_tensor)
        final_tensor = blender_segment(image_tensor, output_tensor.cpu())
    else:
        input_tensor = todos.data.zeropad_tensor(input_tensor, times=SEGMENT_TIMES)
        output_tensor = todos.model.forward(model, device, input_tensor)
        output = blender_tensor[:, :, 0:H, 0:W]
        final_tensor = blender_segment(image_tensor, output_tensor.cpu())

    return final_tensor.unsqueeze(0)


def image_client(name, input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.pmask(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")
    # print(redo)


def image_server(name, HOST="localhost", port=6379):
    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_segment.pth"
    model, device = get_model(checkpoint)

    def do_service(input_file, output_file, targ):
        print(f"  Segment {input_file} ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_pmask", do_service, HOST, port)


def image_predict(input_files, output_dir):
    print(f"Segment predict {input_files} ... ")

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_segment.pth"
    model, device = get_model(checkpoint)

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # input_tensor = load_gray_tensor(filename)

        print(input_tensor.size())

        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()
        predict_tensor = model_forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)


def video_service(input_file, output_file, targ):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    checkpoint = os.path.dirname(__file__) + "/models/image_segment.pth"
    model, device = get_model(checkpoint)

    print(f"  clean {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def clean_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=clean_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_client(name, input_file, output_file):
    cmd = redos.video.Command()
    context = cmd.pmask(input_file, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_pmask", video_service, HOST, port)


def video_predict(input_file, output_file):
    return video_service(input_file, output_file, None)
