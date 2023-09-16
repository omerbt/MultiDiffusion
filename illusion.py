import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerDiscreteScheduler
)
from pipeline_controlnet import StableDiffusionControlNetPipeline
import argparse


# https://huggingface.co/spaces/AP123/IllusionDiffusion/blob/main/app.py
def center_crop_resize(img, output_size=(512, 512)):
    width, height = img.size

    # Calculate dimensions to crop to the center
    new_dimension = min(width, height)
    left = (width - new_dimension) / 2
    top = (height - new_dimension) / 2
    right = (width + new_dimension) / 2
    bottom = (height + new_dimension) / 2

    # Crop and resize
    img = img.crop((left, top, right, bottom))
    img = img.resize(output_size)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--controlnet_img', type=str, default="pano_pattern.png")
    parser.add_argument('--negative_prompt', type=str, default='low quality')
    # controls the fidelity to the controlnet signal. May have to be adjusted depending on the input
    parser.add_argument('--controlnet_scale', type=float, default=1.3)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--H', type=int, default=512)
    parser.add_argument('--W', type=int, default=1536)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--stride', type=int, default=64)
    opt = parser.parse_args()

    h, w = opt.H, opt.W
    h = h - h % opt.stride
    w = w - w % opt.stride

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
    ).to("cuda")

    control_image = Image.open(opt.controlnet_img).convert("RGB")
    control_image = center_crop_resize(control_image, output_size=(w, h))
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    out = pipe(
        prompt=opt.prompt,
        negative_prompt=opt.negative_prompt,
        image=control_image,
        guidance_scale=opt.guidance_scale,
        controlnet_conditioning_scale=opt.controlnet_scale,
        generator=torch.manual_seed(opt.seed) if opt.seed != -1 else torch.Generator(),
        num_inference_steps=opt.steps,
        height=h,
        width=w,
        stride=opt.stride // 8
    ).images[0]
    out.save("out.png")