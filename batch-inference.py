import modal
import numpy as np
import torch
import io
from pathlib import Path
import modal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from modal import App, Image, build, gpu, web_endpoint
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
# from utils.proc import build_transform, find_closest_aspect_ratio, dynamic_preprocess, dynamic_preprocess2, load_image, load_image2 


ocr_gen = (
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
        "pillow",
        "modelscope", 
        "transformers" ,
        "sentencepiece", 
    )
)


app = modal.App("ocr")

@app.cls(gpu="T4", image=ocr_gen)
class Model:
    
    @build()
    def build(self):
        import os
        from huggingface_hub import snapshot_download

        MODEL_DIR = "/ocr"
        MODEL_NAME = "mx262/MiniMonkey"
        MODEL_REVISION = "786ef79f07ae2126a800538bebc279a2bcfe014e"
"

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
        from transformers import AutoModel, AutoTokenizer

        # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        self.id = "mx262/MiniMonkey"
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


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

        # set the max number of tiles in `max_num`
        pixel_values, target_aspect_ratio = load_image( image_data, min_num=4, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        pixel_values2 = load_image2(image_data, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio)
        pixel_values2 = pixel_values2.to(torch.bfloat16).cuda()
        pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

        generation_config = dict(do_sample=False, max_new_tokens=512)

        # question = "extract all text in the image."
        response, history = self.model.chat(self.tokenizer, pixel_values, target_aspect_ratio, prompt, generation_config, history=None, return_history=True)
        # print(f'User: {question} Assistant: {response}')
        return response, history


@app.function(image=ocr_gen,)
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
