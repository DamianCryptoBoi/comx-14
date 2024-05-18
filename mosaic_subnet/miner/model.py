
from io import BytesIO
from typing import Optional
import base64
import threading

import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from communex.module.module import Module, endpoint

from huggingface_hub import hf_hub_download

from transformers import pipeline, set_seed

import re

import random

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

class DiffUsers(Module):
    def __init__(self, model_name: str = "stabilityai/sdxl-turbo") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        ## 2 step lora
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 8

        # prompt
        self.magic_prompt_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

        self._lock = threading.Lock()

    def magic_prompt(self, raw_prompt: str) -> str:
        prompt = raw_prompt.replace("\n", "").lower().capitalize()
        prompt = re.sub(r"[,:\-–.!;?_]", '', prompt)
        generated_text = prompt

        for count in range(5):
            seed = random.randint(100, 1000000)
            set_seed(seed)

            response = self.magic_prompt_pipe(prompt, pad_token_id=50256, max_length=77, truncation=True)[0]
            generated_text = response['generated_text'].strip()
            if len(generated_text) > (len(prompt) + 8):
                if generated_text.endswith((":", "-", "—")) is False:
                    generated_text = re.sub(r'[^ ]+\.[^ ]+', '', generated_text).replace("<", "").replace(">", "")
                break

        return generated_text

    @endpoint
    def sample(
        self, prompt: str, steps: int = 2, negative_prompt: str = "", seed:
    Optional[int]=None) -> str:
        generator = torch.Generator(self.device)
        if seed is None:
            seed = generator.seed()
        generator = generator.manual_seed(seed)
        try:
            magic_prompt = self.magic_prompt(prompt)
        except Exception as e:
            magic_prompt = prompt
        with self._lock:
            image = self.pipeline(
                prompt=magic_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=self.steps,
                generator=generator,
                guidance_scale=5.0
            ).images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    @endpoint
    def get_metadata(self) -> dict:
        return {"model": self.model_name}

if __name__ == "__main__":
    d = DiffUsers()
    out = d.sample(prompt="cat, jumping")
    with open("a.png", "wb") as f:
        f.write(out)
