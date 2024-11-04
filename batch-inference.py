import modal


import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def dynamic_preprocess2(image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio

def load_image2(image_file, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = 'minimonkey'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
pixel_values, target_aspect_ratio = load_image('/content/img (21).jpg', min_num=4, max_num=12)
pixel_values = pixel_values.to(torch.bfloat16).cuda()
pixel_values2 = load_image2('/content/img (21).jpg', min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio)
pixel_values2 = pixel_values2.to(torch.bfloat16).cuda()
pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

generation_config = dict(do_sample=False, max_new_tokens=512)

question = "extract all text in the image."
response, history = model.chat(tokenizer, pixel_values, target_aspect_ratio, question, generation_config, history=None, return_history=True)
print(f'User: {question} Assistant: {response}')



# we will be executing our code using local environment
# Although this can work with Modal _Volumes_.

import modal
from modal import Image

volume = modal.Volume.from_name("vlm-training", create_if_missing=True)

cuda_version = "12.1.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vlm = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "datasets==3.0.1",
        "accelerate==0.34.2",
        "evaluate==0.4.3",
        "bitsandbytes==0.44.0",
        "trl==0.11.1",
        "peft==0.13.0",
        "qwen-vl-utils",
        "python-dotenv",
        "torch~=2.4.0",
        "torchvision",
        "wandb",
        "deepspeed",
        "einops",
        "ujson",
        "decord",
    )
)


app = modal.App("vlm-training", image=vlm)
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
CHECKPOINTS_PATH = "/vol/experiment"



retries = modal.Retries(initial_delay=0.0, max_retries=10)
timeout = 7200  # in seconds this is 2 hrs

@app.function(
    volumes={CHECKPOINTS_PATH: volume},
    gpu=modal.gpu.H100(count=2),
    timeout=timeout, 
    retries=retries
)
def train():
    import os
    import subprocess
    from pathlib import Path
    import sys
    import wandb
    from VLM.utils.data import format2json
    from dotenv import load_dotenv
    from huggingface_hub import login


    # Set up training parameters
    MODEL_NAME = "allenai/Molmo-7B-D-0924"
    DEEPSPEEDPATH = "VLM/utils/zero3.json"






    print("⚡️ Starting training...")

    # Construct the DeepSpeed command
    deepspeed_command = [
        "deepspeed",
        "--num_gpus=2",
        "VLM/src/training/train.py",
        "--lora_enable", "True",
        "--vision_lora", "True",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--lora_dropout", "0.05",
        "--num_lora_modules", "-1",
        "--deepspeed", DEEPSPEEDPATH,
        "--model_id", MODEL_NAME,
        "--data_path", JSON_FILE,
        "--image_folder", IMAGE_FOLDER,
        "--freeze_vision_tower", "False",
        "--freeze_llm", "False",
        "--tune_projector", "True",
        "--bf16", "True",
        "--fp16", "False",
        "--disable_flash_attn2", "False",
        "--output_dir", OUTPUT_DIR,
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", "2e-4",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--gradient_checkpointing", "False",
        "--report_to", "wandb",
        "--lazy_preprocess", "True",
        "--save_strategy", "steps",
        "--save_steps", "200",
        "--save_total_limit", "10",
        "--dataloader_num_workers", "4"
    ]

    # Set the PYTHONPATH to include the necessary directories
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{os.getcwd()}/VLM/src:" + env.get('PYTHONPATH', '')

    # Run the DeepSpeed command
    try:
        result = subprocess.run(
            deepspeed_command,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print('Output:', result.stdout)
    except subprocess.CalledProcessError as e:
        print('Command failed. Return code:', e.returncode)
        print('Output:', e.stdout)
        print('Error:', e.stderr)

    print("Training completed.") 



base_path = Path("/local_dir")

DATASET_NAME = "aidystark/fashion-tag"
OUTPUT_DIR = base_path / "molmo"
JSON_FILE = base_path / "output.json"
IMAGE_FOLDER = base_path / "IMG"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)

HF_TOKEN = "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"
DATASET_ID = "aidystark/fashion-tag"
JSON_FILE = "output.json"
WANDB_APIKEY = "0d505324ba165d96687f3624d4310bf171485b9d"

login(token=HF_TOKEN)
wandb.login(key=WANDB_APIKEY)


# Ensure output directory exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Call the function from data.py to process and format the dataset
format2json(DATASET_NAME, JSON_FILE, IMAGE_FOLDER)


import io
from pathlib import Path
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from modal import App, Image, build, gpu, web_endpoint

caption_gen = (
    Image.debian_slim(python_version="3.10")
    .apt_install("libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "accelerate",
        "bitsandbytes",
        "ray",
        "pillow"
    )
)

app = modal.App("caption-gen")

@app.cls(gpu="T4", image=caption_gen)
class Model:
    
    @build()
    def build(self):
        import os
        from huggingface_hub import snapshot_download

        MODEL_DIR = "/llava"
        MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
        MODEL_REVISION = "a1d521368f8d353afa4da2ed2bb1bf646ef1ff5f"

        os.makedirs(MODEL_DIR, exist_ok=True)

        snapshot_download(
            MODEL_NAME,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        )
    
    @modal.enter()
    def start_engine(self):
        import torch
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        self.model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
        self.processor = LlavaNextProcessor.from_pretrained(self.model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            load_in_4bit=True
        )

    @modal.method()
    async def inference(self, image_data, prompt):
        import torch
        from PIL import Image

        # Loading Image
        def load_img(file_data):
            try:
                image = Image.open(io.BytesIO(file_data)).convert('RGB')
                return image
            except Exception as e:
                raise ValueError("Invalid image data") from e

        image = load_img(await image_data.read())  # Open the image
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda:0")  # Prepare the inputs
        output = self.model.generate(**inputs, max_new_tokens=640)  # Generate output
        decoded_output = self.processor.decode(output[0], skip_special_tokens=True)  # Decode output
        return decoded_output

@app.function(image=caption_gen,)
@web_endpoint(method="POST")
async def img_ttxt(image: UploadFile = File(...), prompt: str = Form(...)):
    try:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")

        #model = Model()
        #caption = await model.inference(image, prompt)
        caption = Model().inference.remote(image, prompt)




        # Return the result
        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        await image.close()
