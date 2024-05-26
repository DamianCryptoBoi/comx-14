from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64
import threading
import time

import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from huggingface_hub import hf_hub_download

from io import BytesIO

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

MAX_CACHE_SIZE = 3

class SampleInput(BaseModel):
    prompt: str
    steps: Optional[int] = 2
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = None

class DiffUsers:
    def __init__(self):
        print("setting up model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # Initialize pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 8

        self._lock = threading.Lock()
        self.cache = {}
        self.cache_order = []
        print("model setup done")

    def _manage_cache(self):
        if len(self.cache_order) >= MAX_CACHE_SIZE:
            oldest_key = self.cache_order.pop(0)
            del self.cache[oldest_key]

    def sample(self, input: SampleInput):
        prompt = input.prompt
        negative_prompt = input.negative_prompt
        seed = input.seed

        cache_key = (prompt, negative_prompt, seed)
        
        with self._lock:
            if cache_key in self.cache:
                return {"image": self.cache[cache_key]}
        
        generator = torch.Generator(self.device)
        if seed is None:
            seed = generator.seed()
        generator = generator.manual_seed(seed)

        with self._lock:
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.steps,
                generator=generator,
                guidance_scale=5
            ).images[0]
        
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode()

        with self._lock:
            self._manage_cache()  # Manage cache size before adding a new item
            self.cache[cache_key] = image_base64
            self.cache_order.append(cache_key)

        return {"image": image_base64}

app = FastAPI()
diffusers = DiffUsers()

@app.post("/sample")
def sample(input: SampleInput):
    return diffusers.sample(input)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
