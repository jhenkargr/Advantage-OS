"""
AdVantage OS - Creative Studio (NVIDIA API Catalog edition)

Dual-mode support:
1. Mounted into main.py via `router`:
   - /image_generation/health  and  /generate/health
   - /image_generation/text2img and  /generate/text2img
   - /image_generation/img2img  and  /generate/img2img
   - /image_generation/caption  and  /generate/caption
2. Standalone execution:
   Run:  uvicorn creative:app --host 0.0.0.0 --port 8000
   Docs: http://127.0.0.1:8000/docs
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

try:
    from creative_studio import nvidia_client as nv
except ImportError:
    import nvidia_client as nv

logger = logging.getLogger(__name__)

if not nv.API_KEY:
    logger.warning(
        "NVIDIA_API_KEY is not set. Image generation routes will return an error until configured."
    )

EXPOSED_HEADERS = [
    "X-Model-Used",
    "X-Seed-Used",
    "X-Img2Img-Method",
    "X-Final-Prompt",
    "X-Warnings",
]

# Compatibility aliases
client = nv
TEXT2IMG_MODEL = nv.IMAGE_MODELS[0] if nv.IMAGE_MODELS else "black-forest-labs/flux.1-dev"
IMG2IMG_MODEL = nv.EDIT_MODEL
CAPTION_MODEL = nv.CHAT_MODELS[0] if nv.CHAT_MODELS else "meta/llama-3.1-8b-instruct"

router = APIRouter(tags=["Image Generation"])


def _header_safe(text: str, limit: int = 500) -> str:
    return " ".join(str(text).split()).encode("ascii", "ignore").decode()[:limit]


def _png_response(image: Image.Image, headers: dict) -> StreamingResponse:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    safe = {k: _header_safe(v) for k, v in headers.items() if v not in (None, "")}
    return StreamingResponse(buf, media_type="image/png", headers=safe)


def _http_error(e: nv.NvidiaError) -> HTTPException:
    status = e.status if e.status in (400, 401, 403, 422, 429) else 502
    return HTTPException(status, str(e))


def image_to_response(image: Image.Image, headers: Optional[dict] = None) -> StreamingResponse:
    """Legacy helper for converting PIL Image to StreamingResponse."""
    return _png_response(image, headers or {})


@router.get("/image_generation/health")
@router.get("/generate/health")
def health():
    """Creative studio status and active model chain details."""
    return {
        "status": "ok",
        "providers": "NVIDIA API Catalog + Groq (captions) + Pollinations (image fallback)",
        "key_configured": bool(nv.API_KEY),
        "text2img_models": nv.IMAGE_MODELS,
        "edit_model": nv.EDIT_MODEL,
        "img2img_mode": nv.IMG2IMG_MODE,
        **nv.status_snapshot(),
    }


@router.post("/image_generation/text2img")
@router.post("/generate/text2img")
def generate_text2img(
    prompt: str = Form(...),
    negative_prompt: Optional[str] = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    """Prompt -> image. width/height snap to NVIDIA supported dimensions (768-1344).
    `negative_prompt` is accepted for compatibility with legacy callers."""
    try:
        image, model, used_seed = nv.text_to_image(prompt, width, height, seed)
    except nv.NvidiaError as e:
        raise _http_error(e)

    headers = {"X-Model-Used": model, "X-Seed-Used": used_seed}
    if negative_prompt:
        headers["X-Negative-Prompt-Note"] = "Ignored by Flux/NVIDIA pipeline"
    return _png_response(image, headers)


@router.post("/image_generation/img2img")
@router.post("/generate/img2img")
def generate_img2img(
    prompt: str = Form(...),
    reference_image: UploadFile = File(...),
    seed: int = Form(-1),
):
    """Reference image + prompt -> transformed image.
    Uses NVIDIA Kontext if available, or vision description + generation pipeline."""
    raw = reference_image.file.read()
    if not raw:
        raise HTTPException(400, "reference_image is empty.")
    try:
        image, info = nv.image_plus_text_to_image(prompt, raw, seed)
    except nv.NvidiaError as e:
        raise _http_error(e)

    return _png_response(image, {
        "X-Model-Used": info.get("model", ""),
        "X-Seed-Used": info.get("seed", -1),
        "X-Img2Img-Method": info.get("method", ""),
        "X-Final-Prompt": info.get("final_prompt", ""),
        "X-Warnings": " | ".join(info.get("warnings", [])),
    })


@router.post("/image_generation/caption")
@router.post("/generate/caption")
def generate_caption(
    description: str = Form(...),
    tone: str = Form("Playful"),
):
    """Description -> short ad caption. Groq / NVIDIA chat with template fallback."""
    try:
        return nv.make_caption(description, tone)
    except nv.NvidiaError as e:
        raise _http_error(e)


# Standalone FastAPI application
app = FastAPI(
    title="AdVantage OS - Creative Studio (NVIDIA)",
    description="Text-to-image, Image-to-image, and Social Caption generation powered by NVIDIA API Catalog & Groq.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=EXPOSED_HEADERS,
)
app.include_router(router)


@app.get("/health", tags=["System"])
def standalone_health():
    return health()