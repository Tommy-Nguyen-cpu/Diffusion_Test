import torch
import secrets
from gradio.networking import setup_tunnel
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    SD3Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
    StableDiffusion3Pipeline,
)

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

isLCM = False
HF_ACCESS_TOKEN = ""

model_path = "segmind/small-sd"
inpaint_model_path = "Lykon/dreamshaper-8-inpainting"
prompt = "Pixar style, a cute hamster in a bonnet, 8k"
promptA = "Pixar style, a cute hamster in a bonnet, 8k"
promptB = "photo realistic, a red fox smoking a pipe, 8k"
negative_prompt = "a photo frame"

num_images = 5
degree = 360
perturbation_size = 0.1
num_inference_steps = 12
seed = 69420

guidance_scale = 8
guidance_values = "1, 8, 20"
intermediate = True
pokeX, pokeY = 256, 256
pokeHeight, pokeWidth = 128, 128
imageHeight, imageWidth = 512, 512

tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder="text_encoder").to(
    torch_device
)

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder = "scheduler")

transformer = SD3Transformer2DModel.from_pretrained(model_path, subfolder="transformer").to(
    torch_device
)
vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(torch_device)

text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_path, subfolder = ("text_encoder_2")).to(torch_device)
tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")


text_encoder_3 = T5EncoderModel.from_pretrained(model_path, subfolder = ("text_encoder_3")).to(torch_device)
tokenizer_3 = T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3")

pipe = StableDiffusion3Pipeline(
    transformer = transformer,
    scheduler = scheduler,
    vae = vae,
    text_encoder = text_encoder,
    tokenizer = tokenizer,
    text_encoder_2 = text_encoder_2,
    tokenizer_2 = tokenizer_2,
    text_encoder_3 = text_encoder_3,
    tokenizer_3 = tokenizer_3
).to(torch_device)

dash_tunnel = setup_tunnel("0.0.0.0", 8000, secrets.token_urlsafe(32))

__all__ = [
    "prompt",
    "negative_prompt",
    "num_images",
    "degree",
    "perturbation_size",
    "num_inference_steps",
    "seed",
    "intermediate",
    "pokeX",
    "pokeY",
    "pokeHeight",
    "pokeWidth",
    "promptA",
    "promptB",
    "tokenizer",
    "text_encoder",
    "scheduler",
    "unet",
    "vae",
    "torch_device",
    "imageHeight",
    "imageWidth",
    "guidance_scale",
    "guidance_values",
    "HF_ACCESS_TOKEN",
    "model_path",
    "inpaint_model_path",
    "dash_tunnel",
    "pipe",
]
