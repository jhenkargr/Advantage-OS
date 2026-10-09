"""
creative.py
-----------
Shared HuggingFace client, model constants, and image helper used by main.py.

Routes are defined in main.py under the /image_generation prefix:
    GET  /image_generation/health
    POST /image_generation/text2img
    POST /image_generation/img2img
    POST /image_generation/caption

Requires HF_TOKEN in .env (https://huggingface.co/settings/tokens).
"""

import io
import os

from fastapi.responses import StreamingResponse
from huggingface_hub import InferenceClient
from PIL import Image

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN not set. Add it to your .env file "
        "(get one free at https://huggingface.co/settings/tokens)."
    )

client = InferenceClient(api_key=HF_TOKEN)

TEXT2IMG_MODEL = "black-forest-labs/FLUX.1-schnell"
IMG2IMG_MODEL  = "black-forest-labs/FLUX.1-Kontext-dev"
CAPTION_MODEL  = "meta-llama/Llama-3.1-8B-Instruct:novita"


def image_to_response(image: Image.Image) -> StreamingResponse:
    """Convert a PIL image to a PNG StreamingResponse."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")