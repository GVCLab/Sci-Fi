import sys
sys.path.append('..')
import argparse
import os

parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
parser.add_argument("--first_image", type=str,required=True, help="The path of the video for controlnet processing.",)
parser.add_argument("--last_image", type=str,required=True, help="The path of the video for controlnet processing.",)

parser.add_argument("--pretrained_model_name_or_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used")
parser.add_argument("--EF_Net_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used")
parser.add_argument("--EF_Net_weights", type=float, default=1.0, help="Strenght of controlnet")
parser.add_argument("--EF_Net_guidance_start", type=float, default=0.0, help="The stage when the controlnet starts to be applied")
parser.add_argument("--EF_Net_guidance_end", type=float, default=1.0, help="The stage when the controlnet end to be applied")

parser.add_argument("--out_path", type=str, default="./output.mp4", help="The path where the generated video will be saved")
parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')")
parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

args = parser.parse_args()

import time
import torch
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from diffusers.utils import export_to_video, load_image 
from Sci_Fi_inbetweening_pipeline import CogVideoXEFNetInbetweeningPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from EF_Net import EF_Net
import cv2
import os
import sys
from decord import VideoReader

@torch.no_grad()
def generate_video(
    prompt: str,
    first_image: str,
    last_image: str,
    pretrained_model_name_or_path: str,
    EF_Net_model_path: str,
    EF_Net_weights: float = 1.0,
    EF_Net_guidance_start: float = 0.0,
    EF_Net_guidance_end: float = 1.0,
    out_path: str = "./output.mp4",
    guidance_scale: float = 6.0,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
):
    """
    Parameters:
    - prompt (str): The description of the video to be generated.
    - first_image (str): The start frame.
    - last_image (str): The end frame.
    - pretrained_model_name_or_path (str): The path of the pre-trained model to be used.
    - transformer_model_path (str): The path of the pre-trained transformer to be used.
    - EF_Net_model_path (str): The path of the pre-trained EF-Net model to be used.
    - EF_Net_weights (float): Strenght of EF-Net
    - EF_Net_guidance_start (float): The stage when the EF-Net starts to be applied
    - EF_Net_guidance_end (float): The stage when the EF-Net end to be applied
    - out_path (str): The path where the generated video will be saved.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    """
    
    # 1. Load the pre-trained CogVideoX-I2V-5B model.
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer")
    vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    scheduler = CogVideoXDDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    
    # 2. Load the pre-trained EF_Net
    EF_Net = EF_Net(num_layers=4, downscale_coef=8, in_channels=2, num_attention_heads=48,).requires_grad_(False).eval()
    ckpt = torch.load(EF_Net_model_path, map_location='cpu', weights_only=False)
    EF_Net_state_dict = {}
    for name, params in ckpt['state_dict'].items():
        EF_Net_state_dict[name] = params
    m, u = EF_Net.load_state_dict(EF_Net_state_dict, strict=False)
    print(f'[ Weights from pretrained EF-Net was loaded into EF-Net ] [M: {len(m)} | U: {len(u)}]')
    
    #3. Load the prompt (Can be modified independently according to specific needs.)
    with open(prompt, 'r', encoding='utf-8') as file:
        prompt = file.read()    
        prompt = prompt.strip()

    # 4. Combine as a pipeline
    pipe = CogVideoXEFNetInbetweeningPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        EF_Net=EF_Net,
        scheduler=scheduler,
    )
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    
    # 5. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    pipe.to("cuda")
    pipe = pipe.to(dtype=dtype)
    #pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 6. Generate the video frames based on the start and end frames, as well as the text prompt
    
    first_image = load_image(first_image)
    last_image = load_image(last_image)
    
    start_time = time.time()
    
    video_generate = pipe(
        first_image=first_image,
        last_image=last_image,
        prompt=prompt,
        num_frames=49,  
        use_dynamic_cfg=False,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
        EF_Net_weights=EF_Net_weights,
        EF_Net_guidance_start=EF_Net_guidance_start,
        EF_Net_guidance_end=EF_Net_guidance_end,
    ).frames[0]
    

    export_to_video(video_generate, out_path, fps=7)
    
    
if __name__ == "__main__":
    
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    generate_video(
        prompt=args.prompt,
        first_image=args.first_image,
        last_image=args.last_image,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        EF_Net_model_path=args.EF_Net_model_path,
        EF_Net_weights=args.EF_Net_weights,
        EF_Net_guidance_start=args.EF_Net_guidance_start,
        EF_Net_guidance_end=args.EF_Net_guidance_end,
        out_path=args.out_path,
        guidance_scale=args.guidance_scale,
        dtype=dtype,
        seed=args.seed,
    )