import asyncio
import base64
import concurrent.futures
import random
import re
import threading
import time
from io import BytesIO

import torch
import uvicorn
from PIL import Image
from communex._common import get_node_url
from communex.client import CommuneClient
from communex.compat.key import classic_load_key
from communex.module.client import ModuleClient
from communex.types import Ss58Address
from diffusers import StableDiffusionXLImg2ImgPipeline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from loguru import logger
from substrateinterface import Keypair
from transformers import pipeline, set_seed

from mosaic_subnet.base import BaseValidator, SampleInput
from mosaic_subnet.base.model import MagicPromptReq
from mosaic_subnet.base.utils import (
    get_netuid,
)
from mosaic_subnet.gateway._config import GatewaySettings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Gateway(BaseValidator):
    def __init__(self, key: Keypair, settings: GatewaySettings) -> None:
        super().__init__()
        self.settings = settings or GatewaySettings()
        self.c_client = CommuneClient(
            get_node_url(use_testnet=self.settings.use_testnet)
        )
        self.key = key
        self.netuid = get_netuid(self.c_client)
        self.call_timeout = self.settings.call_timeout
        self.top_miners = {}
        self.validators: dict[int, tuple[list[str], Ss58Address]] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

        self.sync()

    def sync(self):
        logger.info("fetching top miners...")
        self.top_miners = self.get_top_weights_miners(16)
        logger.info("fetched miners: {}", self.top_miners)

        logger.info("fetching validators...")
        self.validators = self.get_validators()
        logger.info("fetched validators: {}", self.validators)

    def sync_loop(self):
        while True:
            time.sleep(60)
            self.sync()

    def start_sync_loop(self):
        logger.info("start sync loop")
        self._loop_thread = threading.Thread(target=self.sync_loop, daemon=True)
        self._loop_thread.start()

    def get_top_miners(self):
        return self.top_miners

    def get_validator_weights_history(self, validator_info):
        try:
            connection, validator_key = validator_info
            module_ip, module_port = connection
            logger.debug(f"Call {validator_key} - {module_ip}:{module_port}")
            client = ModuleClient(host=module_ip, port=int(module_port), key=self.key)
            result = asyncio.run(
                client.call(
                    fn="get_weights_history",
                    target_key=validator_key,
                    params={},
                    timeout=10,
                )
            )
        except Exception:
            return None
        return result

    def get_all_validators_weights_history(self):
        rv = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            it = executor.map(
                self.get_validator_weights_history, self.validators.values()
            )
            validator_answers = [*it]

        for uid, response in zip(self.validators.keys(), validator_answers):
            rv.append({"uid": uid, "weights_history": response})
        return rv


magic_prompt_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')


@app.post("/magic_prompt")
async def magic_prompt(req: MagicPromptReq):
    prompt = req.prompt.replace("\n", "").lower().capitalize()
    prompt = re.sub(r"[,:\-–.!;?_]", '', prompt)
    generated_text = prompt

    for count in range(5):
        seed = random.randint(100, 1000000)
        set_seed(seed)

        response = magic_prompt_pipe(prompt, pad_token_id=50256, max_length=77, truncation=True)[0]
        generated_text = response['generated_text'].strip()
        if len(generated_text) > (len(prompt) + 8):
            if generated_text.endswith((":", "-", "—")) is False:
                generated_text = re.sub(r'[^ ]+\.[^ ]+', '', generated_text).replace("<", "").replace(">", "")
            break

    return {"text": generated_text}


@app.post(
    "/generate",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def generate_image(req: SampleInput):
    top_miners = list(app.m.get_top_miners().values())
    top_miners = random.sample(top_miners, 5)
    tasks = [
        app.m.get_miner_generation_async(miner_info, req) for miner_info in top_miners
    ]
    for future in asyncio.as_completed(tasks):
        result = await future
        if result:
            result = app.m.pipe(prompt=req.prompt, width=512, height=512, image=Image.open(BytesIO(result))).images[0]
            buffered = BytesIO()
            result.save(buffered, format="PNG")
            return Response(content=buffered.getvalue(), media_type="image/png")
    return Response(content=b"", media_type="image/png")


@app.get("/weights")
async def get_all_validators_weights_history():
    return app.m.get_all_validators_weights_history()


if __name__ == "__main__":
    settings = GatewaySettings(
        host="0.0.0.0",
        port=9009,
        use_testnet=True,
    )
    app.m = Gateway(key=classic_load_key("mosaic-validator0"), settings=settings)
    app.m.start_sync_loop()
    uvicorn.run(app=app, host=settings.host, port=settings.port)
