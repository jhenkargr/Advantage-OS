"""
Provider layer for Creative Studio (file keeps its name: NVIDIA is the main provider).

  Images   : NVIDIA FLUX.1-dev -> NVIDIA FLUX.1-schnell -> Pollinations (free fallback)
  Captions : Groq (free, optional key) -> NVIDIA chat models -> template
  Img+text : NVIDIA Kontext if it accepts your image, else vision-model description -> image chain above

Reliability features (so it keeps working on busy, shared free tiers):
  * reads YOUR NVIDIA key's live model list and skips retired models
  * a model/provider that hangs, errors or is retired is skipped for a while
  * hard time budgets, short timeouts, retry with back-off on 429/5xx
  * polling of NVIDIA's 202 "queued" jobs (without resubmitting them)
  * safety: if NVIDIA's content filter blocks a prompt we STOP (never retried on another provider)
  * captions never fail: template fallback
"""

import base64
import io
import os
import random
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
POLLINATIONS_FALLBACK = os.getenv("POLLINATIONS_FALLBACK", "1").strip() != "0"

IMAGE_BASE = "https://ai.api.nvidia.com/v1/genai"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _csv_env(name: str, default: List[str]) -> List[str]:
    items = [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]
    return items or default


# Tried in this order. Override any of them in .env (comma separated).
IMAGE_MODELS = _csv_env(
    "NVIDIA_IMAGE_MODELS",
    ["black-forest-labs/flux.1-dev", "black-forest-labs/flux.1-schnell"],
)
EDIT_MODEL = os.getenv("NVIDIA_EDIT_MODEL", "black-forest-labs/flux.1-kontext-dev")
CHAT_MODELS = _csv_env(
    "NVIDIA_CHAT_MODELS",
    [
        "qwen/qwen3-next-80b-a3b-instruct",
        "meta/llama-3.2-3b-instruct",
        "mistralai/ministral-14b-instruct-2512",
        "microsoft/phi-4-mini-instruct",
        "mistralai/mistral-7b-instruct-v0.3",
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.3-70b-instruct",
    ],
)
VISION_MODELS = _csv_env(
    "NVIDIA_VISION_MODELS",
    [
        "meta/llama-3.2-11b-vision-instruct",
        "meta/llama-3.2-90b-vision-instruct",
        "google/gemma-3-27b-it",
        "google/gemma-3n-e4b-it",
        "microsoft/phi-4-multimodal-instruct",
    ],
)
GROQ_MODELS = _csv_env("GROQ_MODELS", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"])

# auto     = try true Kontext editing with the user's image first, fall back to describe-and-generate
# describe = always describe-and-generate (skips the Kontext attempt)
# kontext  = Kontext only, error if it is not accepted
IMG2IMG_MODE = os.getenv("NVIDIA_IMG2IMG_MODE", "auto").strip().lower()

IMAGE_TIMEOUT = int(os.getenv("NVIDIA_IMAGE_TIMEOUT", "75"))    # seconds before giving up on one image call
IMAGE_BUDGET = int(os.getenv("NVIDIA_IMAGE_BUDGET", "150"))     # total seconds for all NVIDIA image tries
CHAT_TIMEOUT = int(os.getenv("NVIDIA_CHAT_TIMEOUT", "30"))
CHAT_BUDGET = int(os.getenv("NVIDIA_CHAT_BUDGET", "45"))
CHAT_MAX_TRIES = 8

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
ALLOWED_SIZES = [768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344]

# Remembers (per server run) whether NVIDIA's hosted Kontext accepted a custom image.
# None = not tried yet, True = works, False = rejected -> skip straight to the fallback.
_kontext_custom_ok: Optional[bool] = None


class NvidiaError(Exception):
    def __init__(self, message: str, status: int = 502, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class ContentFiltered(NvidiaError):
    """NVIDIA's safety filter blocked the prompt or the output."""

    def __init__(self, message: str = "NVIDIA's safety filter blocked this request. Try rephrasing the prompt."):
        super().__init__(message, status=422, retryable=False)


def _scrub(text: str) -> str:
    """Never let a secret key leak into an error message."""
    for secret in (API_KEY, GROQ_API_KEY, POLLINATIONS_API_KEY):
        if secret:
            text = text.replace(secret, "***")
    return text


# ----------------------------------------------------------------------
# Model discovery + health tracking
# ----------------------------------------------------------------------

_live_ids: Optional[set] = None
_live_checked_at = 0.0
_bad_until: Dict[str, float] = {}


def live_models(force: bool = False) -> Optional[set]:
    """Model ids your NVIDIA key can see right now (cached 30 min). None if NVIDIA can't be reached.
    Note: 'visible' does not always mean 'callable' - failures are handled by the fallbacks."""
    global _live_ids, _live_checked_at
    if not force and _live_ids is not None and time.time() - _live_checked_at < 1800:
        return _live_ids
    try:
        r = requests.get(MODELS_URL, headers=_headers(), timeout=20)
        if r.status_code == 200:
            _live_ids = {m["id"] for m in r.json().get("data", [])}
            _live_checked_at = time.time()
    except Exception:
        pass
    return _live_ids


def _is_bad(key: str) -> bool:
    return _bad_until.get(key, 0) > time.time()


def _mark_bad(key: str, seconds: int) -> None:
    _bad_until[key] = time.time() + seconds


def _mark_good(key: str) -> None:
    _bad_until.pop(key, None)


def _note_failure(key: str, err: "NvidiaError") -> None:
    """Retired/missing models are skipped for an hour; hung/erroring ones for 5 minutes.
    Plain request-format rejections (400/422) don't count against a model."""
    if err.status in (404, 410):
        _mark_bad(key, 3600)
    elif err.retryable or err.status >= 500:
        _mark_bad(key, 300)


def _order(models: List[str], prefix: str = "") -> List[str]:
    """Healthy models first; recently-failed ones last (still tried if nothing else works)."""
    return ([m for m in models if not _is_bad(prefix + m)] +
            [m for m in models if _is_bad(prefix + m)])


_SKIP_WORDS = ("guard", "safety", "embed", "rerank", "reason", "thinking", "parse", "coder", "gliner",
               "clip", "vision", "-vl", "content", "detect", "translate", "jailbreak", "topic", "pii")
_FAMILY_RANK = {"qwen": 0, "meta": 1, "mistralai": 2, "google": 3, "microsoft": 4, "nvidia": 5}


def _auto_pick_chat(ids: set) -> List[str]:
    """Extra plain instruct models from your live list (known families first)."""
    cands = [m for m in ids if "instruct" in m.lower() and not any(w in m.lower() for w in _SKIP_WORDS)]
    cands.sort(key=lambda m: (_FAMILY_RANK.get(m.split("/")[0].lower(), 9), m))
    return cands[:3]


def chat_chain() -> List[str]:
    """NVIDIA chat models to try for captions: preferred (if live) + a few auto-picked extras,
    then vision-capable models as a last text fallback (they also answer plain text)."""
    ids = live_models()
    if ids is None:
        base = list(CHAT_MODELS)
        tail = list(VISION_MODELS[:2])
    else:
        base = [m for m in CHAT_MODELS if m in ids]
        base += [m for m in _auto_pick_chat(ids) if m not in base]
        tail = [m for m in VISION_MODELS if m in ids][:2]
    tail = [m for m in tail if m not in base]
    return _order(base) + _order(tail)


def chat_entries() -> List[Tuple[str, str]]:
    """(provider, model) pairs in the order captions are attempted."""
    entries: List[Tuple[str, str]] = []
    if GROQ_API_KEY and not _is_bad("groq:*"):
        entries += [("groq", m) for m in _order(GROQ_MODELS, "groq:")]
    entries += [("nvidia", m) for m in chat_chain()]
    return entries


def vision_chain() -> List[str]:
    ids = live_models()
    if ids is None:
        return _order(VISION_MODELS)
    live = [m for m in VISION_MODELS if m in ids]
    return _order(live) if live else list(VISION_MODELS)


def image_chain() -> List[str]:
    return _order(IMAGE_MODELS)


def status_snapshot() -> dict:
    now = time.time()
    return {
        "live_model_list_loaded": live_models() is not None,
        "image_chain": image_chain() + (["pollinations (fallback)"] if POLLINATIONS_FALLBACK else []),
        "caption_chain": [f"{p}:{m}" for p, m in chat_entries()],
        "vision_chain": vision_chain(),
        "groq_configured": bool(GROQ_API_KEY),
        "pollinations_key_configured": bool(POLLINATIONS_API_KEY),
        "temporarily_skipped": {m: int(t - now) for m, t in _bad_until.items() if t > now},
        "kontext_accepts_custom_images": _kontext_custom_ok,
    }


# ----------------------------------------------------------------------
# Low-level HTTP with retries
# ----------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _groq_headers() -> dict:
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _error_from(resp: requests.Response, who: str = "NVIDIA") -> NvidiaError:
    try:
        body = resp.json()
        detail = body.get("detail") or body.get("error") or body.get("message") or body
    except Exception:
        detail = resp.text[:300]
    detail = _scrub(str(detail)[:400])

    if resp.status_code in (401, 403):
        where = "console.groq.com/keys" if who == "Groq" else "build.nvidia.com/settings/api-keys"
        return NvidiaError(
            f"{who} rejected the API key ({resp.status_code}). Generate a new key at {where} "
            f"and update .env. Detail: {detail}",
            status=resp.status_code,
        )
    if resp.status_code == 410:
        return NvidiaError(f"Model retired by {who} (410): {detail}", status=410)
    if resp.status_code == 404:
        return NvidiaError(f"Model/endpoint not found or not available for this key: {detail}", status=404)
    if resp.status_code == 429:
        return NvidiaError(f"{who} rate limit hit (free tier is per-minute limited).", status=429, retryable=True)
    if resp.status_code in RETRYABLE_STATUS:
        return NvidiaError(f"{who} temporary error {resp.status_code}: {detail}", status=resp.status_code, retryable=True)
    return NvidiaError(f"{who} rejected the request ({resp.status_code}): {detail}", status=resp.status_code)


def _poll(request_id: Optional[str], timeout: int) -> requests.Response:
    """NVIDIA answers 202 when a job is queued/running; poll until it finishes.
    On timeout we do NOT resubmit (that would put the job at the back of the queue again);
    the caller moves on to the next model instead."""
    if not request_id:
        raise NvidiaError("NVIDIA returned 202 without a request id to poll.", status=502)
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = requests.get(STATUS_URL + request_id, headers=_headers(), timeout=30)
        if resp.status_code != 202:
            return resp
        time.sleep(2)
    raise NvidiaError(
        "Timed out waiting in NVIDIA's queue (the shared free tier is busy for this model).",
        status=504, retryable=False,
    )


def _invoke(url: str, payload: dict, timeout: int = 75, attempts: int = 2,
            headers: Optional[dict] = None, who: str = "NVIDIA",
            deadline: Optional[float] = None) -> dict:
    if headers is None:
        if not API_KEY:
            raise NvidiaError("NVIDIA_API_KEY is not set. Put it in your .env file.", status=500)
        headers = _headers()

    last: Optional[NvidiaError] = None
    for attempt in range(attempts):
        if attempt > 0 and deadline is not None and time.time() + 10 > deadline:
            break  # not enough time budget left for another try
        this_timeout = timeout if deadline is None else max(5, min(timeout, int(deadline - time.time())))
        delay = 2 * (2 ** attempt) + random.random()  # 2s, 4s, 8s ... (+ jitter)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=this_timeout)
            if resp.status_code == 202:
                resp = _poll(resp.headers.get("NVCF-REQID"), this_timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = NvidiaError(_scrub(f"Network problem talking to {who}: {e}"), status=503, retryable=True)
        except NvidiaError as e:
            last = e
            if not e.retryable:
                raise
        else:
            if resp.status_code == 200:
                return resp.json()
            err = _error_from(resp, who)
            if not err.retryable:
                raise err
            last = err
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = min(int(retry_after), 30)

        if attempt < attempts - 1:
            time.sleep(delay)

    raise last or NvidiaError(f"{who} request failed.", status=502)


# ----------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------

def _snap(value: int) -> int:
    return min(ALLOWED_SIZES, key=lambda s: abs(s - int(value)))


def _extract_image(data: dict) -> Image.Image:
    art = (data.get("artifacts") or [None])[0]
    b64 = None
    if art:
        reason = art.get("finishReason")
        if reason == "CONTENT_FILTERED":
            raise ContentFiltered()
        if reason and reason != "SUCCESS":
            raise NvidiaError(f"NVIDIA finished with status '{reason}'.", status=502)
        b64 = art.get("base64")
    b64 = b64 or data.get("image") or ((data.get("data") or [{}])[0].get("b64_json"))
    if not b64:
        raise NvidiaError(f"NVIDIA response contained no image (keys: {list(data.keys())}).", status=502)
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _image_payload(model: str, prompt: str, width: int, height: int, seed: int) -> dict:
    payload = {"prompt": prompt, "width": width, "height": height, "seed": seed}
    if "schnell" in model:
        payload["steps"] = 4            # schnell is distilled for ~4 steps
    else:
        payload["steps"] = 30           # flux.1-dev: quality/speed balance
        payload["cfg_scale"] = 3.5
    return payload


def _generate(model: str, prompt: str, width: int, height: int, seed: int,
              timeout: int, attempts: int, deadline: Optional[float] = None) -> Image.Image:
    data = _invoke(f"{IMAGE_BASE}/{model}", _image_payload(model, prompt, width, height, seed),
                   timeout=timeout, attempts=attempts, deadline=deadline)
    return _extract_image(data)


def _pollinations_image(prompt: str, width: int, height: int, seed: int, timeout: int = 60) -> Tuple[Image.Image, str]:
    """Free fallback image provider. Works without a key (anonymous tier: throttled, may be watermarked);
    a free key from enter.pollinations.ai lifts both limits."""
    params = {"model": "flux", "width": width, "height": height, "seed": seed, "nologo": "true"}
    encoded = quote(prompt[:900], safe="")
    if POLLINATIONS_API_KEY:
        url, label = f"https://gen.pollinations.ai/image/{encoded}", "pollinations/flux"
        params["key"] = POLLINATIONS_API_KEY
    else:
        url, label = f"https://image.pollinations.ai/prompt/{encoded}", "pollinations/flux (anonymous)"
    try:
        r = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise NvidiaError(_scrub(f"Pollinations network problem: {e}"), status=503, retryable=True)
    if r.status_code == 429:
        raise NvidiaError("Pollinations anonymous tier is throttled (about 1 request / 15 s). "
                          "Retry shortly or add a free POLLINATIONS_API_KEY.", status=429, retryable=True)
    if r.status_code != 200 or not str(r.headers.get("Content-Type", "")).startswith("image/"):
        raise NvidiaError(f"Pollinations returned status {r.status_code}.", status=502)
    try:
        return Image.open(io.BytesIO(r.content)).convert("RGB"), label
    except Exception:
        raise NvidiaError("Pollinations returned data that is not a valid image.", status=502)


def compress_for_upload(raw: bytes, limit_chars: int = 150_000) -> str:
    """Shrink an uploaded image to a base64 JPEG small enough for NVIDIA's
    inline-image limit (~180KB). Returns the base64 string (no data: prefix)."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    b64 = ""
    for max_side, quality in [(768, 80), (640, 75), (512, 70), (384, 65), (256, 60)]:
        im = img.copy()
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        if len(b64) <= limit_chars:
            break
    return b64


def _size_like(raw: bytes) -> Tuple[int, int]:
    """Pick a generation size that keeps the reference image's orientation."""
    w, h = Image.open(io.BytesIO(raw)).size
    ratio = w / h if h else 1
    if ratio > 1.3:
        return 1344, 768
    if ratio < 0.77:
        return 768, 1344
    return 1024, 1024


# ----------------------------------------------------------------------
# 1) Text -> image
# ----------------------------------------------------------------------

def text_to_image(prompt: str, width: int = 1024, height: int = 1024, seed: int = -1) -> Tuple[Image.Image, str, int]:
    """Returns (image, model_used, seed_used).
    NVIDIA models first (healthiest first, within a time budget), then the free Pollinations fallback."""
    width, height = _snap(width), _snap(height)
    if seed < 0:
        seed = random.randint(1, 2_147_483_647)

    errors = []
    deadline = time.time() + IMAGE_BUDGET
    for i, model in enumerate(image_chain()):
        remaining = deadline - time.time()
        if remaining < 20:
            errors.append(f"{model}: skipped (time budget used up)")
            break
        try:
            image = _generate(model, prompt, width, height, seed,
                              timeout=min(IMAGE_TIMEOUT, int(remaining)),
                              attempts=2 if i == 0 else 1, deadline=deadline)
            _mark_good(model)
            return image, model, seed
        except ContentFiltered:
            raise  # never work around a safety block by asking another provider
        except NvidiaError as e:
            if e.status in (401, 403):
                raise  # bad NVIDIA key: a configuration problem the developer must see
            _note_failure(model, e)
            errors.append(f"{model}: {e}")

    if POLLINATIONS_FALLBACK:
        try:
            image, label = _pollinations_image(prompt, width, height, seed)
            return image, label, seed
        except NvidiaError as e:
            errors.append(f"pollinations: {e}")
    raise NvidiaError("All image providers failed -> " + " | ".join(errors), status=502)


# ----------------------------------------------------------------------
# 2) Chat + caption
# ----------------------------------------------------------------------

def chat(messages: list, max_tokens: int = 120, temperature: float = 0.7,
         models: Optional[List[str]] = None) -> Tuple[str, str]:
    """Returns (text, label). `models` (optional) forces specific NVIDIA models (used by the vision step);
    otherwise Groq (if configured) then NVIDIA models are tried, within a time budget."""
    entries = [("nvidia", m) for m in models] if models is not None else chat_entries()
    entries = entries[:CHAT_MAX_TRIES]
    deadline = time.time() + CHAT_BUDGET
    errors = []
    last_err: Optional[NvidiaError] = None

    for provider, model in entries:
        if time.time() > deadline:
            errors.append("time budget used up")
            break
        if provider == "groq" and _is_bad("groq:*"):
            continue  # Groq key was rejected earlier (maybe in this very request): skip all Groq models
        label = model if provider == "nvidia" else f"groq/{model}"
        health_key = model if provider == "nvidia" else f"groq:{model}"
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "stream": False}
        try:
            if provider == "groq":
                data = _invoke(GROQ_URL, payload, timeout=CHAT_TIMEOUT, attempts=2,
                               headers=_groq_headers(), who="Groq", deadline=deadline)
            else:
                data = _invoke(CHAT_URL, payload, timeout=CHAT_TIMEOUT, attempts=2, deadline=deadline)
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if text:
                _mark_good(health_key)
                return text, label
            errors.append(f"{label}: empty reply")
        except NvidiaError as e:
            if e.status in (401, 403):
                if provider == "nvidia":
                    raise
                _mark_bad("groq:*", 3600)   # bad Groq key: stop using Groq, carry on with NVIDIA
                errors.append(f"{label}: {e}")
                continue
            _note_failure(health_key, e)
            errors.append(f"{label}: {e}")
            last_err = e
        except (KeyError, IndexError, TypeError):
            errors.append(f"{label}: unexpected response shape")

    if last_err is not None and len(entries) == 1:
        raise last_err  # single-model call: keep the real status (400/410/...) for the caller
    raise NvidiaError("All chat models failed -> " + " | ".join(errors), status=502)


CAPTION_FALLBACKS = {
    "playful": "✨ {d} — come try it before it's gone! 😋",
    "elegant": "Discover {d}, thoughtfully crafted for you. ✨",
    "urgent": "🔥 {d} — limited time only, don't miss out!",
}


def _clean_line(text: str) -> str:
    """First non-empty line, with wrapping quotes removed."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[0].strip('"').strip("'").strip("\u201c\u201d").strip()


def make_caption(description: str, tone: str = "Playful") -> dict:
    """Never raises (except for a bad NVIDIA key): falls back to a template if every LLM call fails."""
    messages = [
        {"role": "system", "content": "You write short, catchy social-media ad captions."},
        {"role": "user", "content": (
            f"Write ONE {tone.lower()}-tone Instagram ad caption (max 20 words) about: "
            f"{description}. Include exactly one relevant emoji. Reply with ONLY the caption text."
        )},
    ]
    try:
        text, label = chat(messages, max_tokens=80, temperature=0.8)
        caption = _clean_line(text)
        if caption:
            return {"caption": caption, "source": "llm", "model": label}
        warning = "LLM returned an empty caption"
    except NvidiaError as e:
        if e.status in (401, 403):
            raise  # config problem the developer must see, not a template
        warning = str(e)

    key = "urgent" if tone.lower().startswith("urgent") else tone.lower()
    template = CAPTION_FALLBACKS.get(key, CAPTION_FALLBACKS["playful"])
    return {"caption": template.format(d=description), "source": "template", "warning": warning}


# ----------------------------------------------------------------------
# 3) Image + text -> image
# ----------------------------------------------------------------------

def _vision_content_openai(instruction: str, b64: str) -> list:
    return [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]


def _vision_content_inline(instruction: str, b64: str) -> str:
    return f'{instruction} <img src="data:image/jpeg;base64,{b64}" />'


def describe_and_merge(user_prompt: str, b64: str) -> str:
    """Ask a vision-language model to look at the reference image and write one
    generation prompt that keeps its subject/style while applying the user's change."""
    instruction = (
        "You are writing a prompt for an AI image generator. "
        f'The user wants: "{user_prompt}". '
        "Look at the reference image. Write ONE image-generation prompt (max 70 words) that keeps the "
        "reference image's main subject, colors, style, lighting and composition while applying the "
        "user's request. Reply with ONLY the prompt text."
    )
    errors = []
    for model in vision_chain()[:3]:
        for build in (_vision_content_openai, _vision_content_inline):
            try:
                text, _ = chat([{"role": "user", "content": build(instruction, b64)}],
                               max_tokens=180, temperature=0.4, models=[model])
                prompt = text.strip().strip('"')
                for prefix in ("Prompt:", "prompt:"):
                    if prompt.startswith(prefix):
                        prompt = prompt[len(prefix):].strip()
                if prompt:
                    return prompt[:700]
            except NvidiaError as e:
                errors.append(f"{model}: {e}")
                if e.status in (401, 403):
                    raise
                if e.status in (404, 410) or e.retryable or e.status >= 500:
                    break  # this model is gone/busy; don't try its second format
    raise NvidiaError("No vision model could read the reference image -> " + " | ".join(errors[-3:]), status=502)


def edit_with_kontext(prompt: str, b64: str, seed: int) -> Image.Image:
    payload = {
        "prompt": prompt,
        "image": f"data:image/jpeg;base64,{b64}",
        "aspect_ratio": "match_input_image",
        "cfg_scale": 3.5,
        "steps": 30,
        "seed": seed,
    }
    data = _invoke(f"{IMAGE_BASE}/{EDIT_MODEL}", payload, timeout=IMAGE_TIMEOUT, attempts=1)
    return _extract_image(data)


def image_plus_text_to_image(prompt: str, raw_image: bytes, seed: int = -1) -> Tuple[Image.Image, dict]:
    """Returns (image, info). info['method'] is 'kontext' (true image editing) or
    'describe-and-generate' (reference described by a vision model, then generated)."""
    global _kontext_custom_ok

    try:
        b64 = compress_for_upload(raw_image)
    except Exception:
        raise NvidiaError("Could not read the reference image - is it a valid image file?", status=400)

    if seed < 0:
        seed = random.randint(1, 2_147_483_647)
    warnings: List[str] = []

    try_kontext = IMG2IMG_MODE == "kontext" or (
        IMG2IMG_MODE == "auto" and _kontext_custom_ok is not False and not _is_bad(EDIT_MODEL)
    )
    if try_kontext:
        try:
            image = edit_with_kontext(prompt, b64, seed)
            _kontext_custom_ok = True
            return image, {"method": "kontext", "model": EDIT_MODEL, "seed": seed,
                           "final_prompt": prompt, "reference_used": True, "warnings": warnings}
        except ContentFiltered:
            raise
        except NvidiaError as e:
            if e.status in (401, 403):
                raise
            if e.status in (400, 404, 410, 422):
                _kontext_custom_ok = False   # hosted preview doesn't accept custom images
            else:
                _mark_bad(EDIT_MODEL, 600)   # busy/hung: skip it for a while
            warnings.append(f"Kontext editing unavailable ({e.status}); used describe-and-generate instead.")
            if IMG2IMG_MODE == "kontext":
                raise

    width, height = _size_like(raw_image)
    reference_used = True
    try:
        final_prompt = describe_and_merge(prompt, b64)
    except NvidiaError as e:
        if e.status in (401, 403):
            raise
        final_prompt = prompt
        reference_used = False
        warnings.append(f"Could not analyse the reference image, used your text prompt only: {e}")

    image, model, used_seed = text_to_image(final_prompt, width, height, seed)
    return image, {"method": "describe-and-generate", "model": model, "seed": used_seed,
                   "final_prompt": final_prompt, "reference_used": reference_used, "warnings": warnings}


def kontext_custom_images_supported() -> Optional[bool]:
    return _kontext_custom_ok
