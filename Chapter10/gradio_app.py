import torch
from diffusers import StableDiffusionXLPipeline

sd_pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16)

sd_pipe.to('cuda')


def generate_image(
        prompt: str,
        inf_steps: int = 100):
    return sd_pipe(
        prompt=prompt,
        num_inference_steps=inf_steps).images[0]


import gradio as gr

interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.components.Textbox(label='Prompt'),
        gr.components.Slider(
            minimum=0,
            maximum=100,
            label='Inference Steps')],
    outputs=gr.components.Image(),
    title='Stable Diffusion',
)

interface.launch()
