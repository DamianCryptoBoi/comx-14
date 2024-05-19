from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional
import base64
import threading

import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from communex.module.module import Module, endpoint

from huggingface_hub import hf_hub_download

from io import BytesIO

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

class SampleInput(BaseModel):
    prompt: str
    steps: Optional[int] = 2
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = None

class DiffUsers:
    def __init__(self):

        print("setting up model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        ## 2 step lora
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 8

        self._lock = threading.Lock()
        print("model setup done")

    def sample(self, input: SampleInput):
        prompt = input.prompt
        # steps = input.steps
        negative_prompt = input.negative_prompt
        seed = input.seed

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
                guidance_scale=5.0
            ).images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        image = base64.b64encode(buf.read()).decode()
        return {"image": image}


app = FastAPI()
diffusers = DiffUsers()


@app.post("/sample")
def sample(input: SampleInput):
    return diffusers.sample(input)

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app=app, host="0.0.0.0", port=8000)