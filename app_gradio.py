"""
Adapted from https://huggingface.co/spaces/stabilityai/stable-diffusion
"""

import torch

import time

import gradio as gr

from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)
# pipe = StableDiffusionPanoramaPipeline.from_pretrained(
#      model_ckpt, scheduler=scheduler
# )

pipe = pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def generate_image_fn(prompt: str, img_width: int, img_height=512) -> list:
    start_time = time.time()
    image = pipe(prompt, height=img_height, width=img_width).images
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
    return image


description = """This Space demonstrates MultiDiffusion Text2Panorama using Stable Diffusion model. To get started, either enter a prompt or pick one from the examples below. For details, please visit [the project page](https://multidiffusion.github.io/).
        <p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
        <br/>
        <a href="https://huggingface.co/spaces/weizmannscience/MultiDiffusion?duplicate=true">
        <img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
        <p/>"""
article = "This Space leverages a A10 GPU to run the predictions. We use mixed-precision to speed up the inference latency."
gr.Interface(
    generate_image_fn,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            max_lines=1,
            placeholder="a photo of the dolomites",
        ),
        gr.Slider(value=4096, minimum=512, maximum=4608, step=128),
    ],
    outputs=gr.Gallery().style(grid=[2], height="auto"),
    title="Generate a panoramic image!",
    description=description,
    article=article,
    examples=[["a photo of the dolomites", 4096]],
    allow_flagging=False,
).launch(enable_queue=True)