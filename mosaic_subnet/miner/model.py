
from io import BytesIO
from typing import Optional
import base64
import threading
import requests

# import torch
# from diffusers import DiffusionPipeline, DDIMScheduler

from communex.module.module import Module, endpoint

# from huggingface_hub import hf_hub_download

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

class DiffUsers(Module):
    def __init__(self, model_name: str = "stabilityai/sdxl-turbo") -> None:
        super().__init__()
        # self.model_name = model_name
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        # ## 2 step lora
        # self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        # self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        # self.pipeline.fuse_lora()
        # self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        # self.steps = 8

        # self._lock = threading.Lock()

    @endpoint
    def sample(
        self, prompt: str, steps: int = 2, negative_prompt: str = "", seed:
    Optional[int]=None) -> str:
            url = "http://0.0.0.0:8000/sample"
            data = {
                "prompt": prompt,
                "steps": steps,
                "negative_prompt": negative_prompt,
                "seed": seed
            }
            response = requests.post(url, json=data)
            return response.json()["image"]

    @endpoint
    def get_metadata(self) -> dict:
        return {"model": self.model_name}

if __name__ == "__main__":
    d = DiffUsers()
    out = d.sample(prompt="cat, jumping")
    with open("a.png", "wb") as f:
        f.write(out)
