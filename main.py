"""
AdVantage OS - Creative Studio (NVIDIA API Catalog edition)

Three independent tools:
    POST /generate/text2img   prompt                      -> PNG image
    POST /generate/img2img    prompt + reference_image    -> PNG image
    POST /generate/caption    description + tone          -> JSON caption

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
Docs: http://127.0.0.1:8000/docs   (interactive page to test every endpoint)
"""

import io

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

import nvidia_client as nv

if not nv.API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. Copy .env.example to .env and paste your key "
        "(get one free at https://build.nvidia.com/settings/api-keys)."
    )

EXPOSED_HEADERS = ["X-Model-Used", "X-Seed-Used", "X-Img2Img-Method", "X-Final-Prompt", "X-Warnings"]

app = FastAPI(title="AdVantage OS - Creative Studio (NVIDIA)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # tighten to your frontend's URL before deploying
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=EXPOSED_HEADERS,
)


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "providers": "NVIDIA API Catalog + Groq (captions) + Pollinations (image fallback)",
        "key_configured": bool(nv.API_KEY),
        "img2img_mode": nv.IMG2IMG_MODE,
        **nv.status_snapshot(),   # model chains in use right now, skipped models, Kontext status
    }


@app.post("/generate/text2img")
def generate_text2img(
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    """Prompt -> image. width/height snap to NVIDIA's supported sizes (768-1344)."""
    try:
        image, model, used_seed = nv.text_to_image(prompt, width, height, seed)
    except nv.NvidiaError as e:
        raise _http_error(e)
    return _png_response(image, {"X-Model-Used": model, "X-Seed-Used": used_seed})


@app.post("/generate/img2img")
def generate_img2img(
    prompt: str = Form(...),
    reference_image: UploadFile = File(...),
    seed: int = Form(-1),
):
    """Reference image + prompt -> image. See README for how the two methods work.
    (Plain `def`, not `async def`: the NVIDIA calls block, so FastAPI runs this in a
    worker thread and the server stays responsive for other requests.)"""
    raw = reference_image.file.read()
    if not raw:
        raise HTTPException(400, "reference_image is empty.")
    try:
        image, info = nv.image_plus_text_to_image(prompt, raw, seed)
    except nv.NvidiaError as e:
        raise _http_error(e)
    return _png_response(image, {
        "X-Model-Used": info["model"],
        "X-Seed-Used": info["seed"],
        "X-Img2Img-Method": info["method"],
        "X-Final-Prompt": info["final_prompt"],
        "X-Warnings": " | ".join(info["warnings"]),
    })


@app.post("/generate/caption")
def generate_caption(
    description: str = Form(...),
    tone: str = Form("Playful"),
):
    """Description -> short ad caption. Always returns a caption (template fallback)."""
    return nv.make_caption(description, tone)
