"""
Run this FIRST:   python check_setup.py

Tests your keys and every feature the backend uses, exactly the way the server uses them
(including the automatic fallbacks), and tells you which provider answered each request.
If something fails, copy the whole output and send it to your guide.
Creates check_text2img.png and check_img2img.png so you can look at the results.
"""

import sys
import time

import requests
from PIL import Image, ImageDraw

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles + emoji
except Exception:
    pass

import nvidia_client as nv

results = []


def step(name, fn):
    print(f"\n--- {name}")
    start = time.time()
    try:
        detail = fn()
        print(f"[PASS] {detail}  ({time.time() - start:.1f}s)")
        results.append((name, True))
    except Exception as e:
        print(f"[FAIL] {e}  ({time.time() - start:.1f}s)")
        results.append((name, False))


def check_keys():
    if not nv.API_KEY:
        raise RuntimeError("NVIDIA_API_KEY missing - create a .env file (copy .env.example) and paste your key.")
    if not nv.API_KEY.startswith("nvapi-"):
        raise RuntimeError("NVIDIA key should start with 'nvapi-'. Re-copy it from build.nvidia.com/settings/api-keys.")
    r = requests.get(nv.MODELS_URL, headers=nv._headers(), timeout=30)
    if r.status_code in (401, 403):
        raise RuntimeError(f"NVIDIA rejected the key ({r.status_code}). Generate a new one.")
    r.raise_for_status()
    nv.live_models(force=True)
    snap = nv.status_snapshot()
    print("  images  :", snap["image_chain"])
    print("  captions:", snap["caption_chain"])
    print("  vision  :", snap["vision_chain"])
    notes = []
    if not nv.GROQ_API_KEY:
        notes.append("tip: add a free GROQ_API_KEY (console.groq.com) for faster, more reliable captions")
    elif not nv.GROQ_API_KEY.startswith("gsk_"):
        notes.append("warning: GROQ_API_KEY should start with 'gsk_'")
    if not nv.POLLINATIONS_API_KEY:
        notes.append("tip: an optional free POLLINATIONS_API_KEY (enter.pollinations.ai) improves the image fallback")
    return "NVIDIA key accepted" + ("".join(f"\n         {n}" for n in notes))


def check_text2img():
    img, model, seed = nv.text_to_image("a red apple on a wooden table, studio photo", 768, 768)
    img.save("check_text2img.png")
    skipped = nv.status_snapshot()["temporarily_skipped"]
    extra = f" | models skipped after errors: {list(skipped)}" if skipped else ""
    return f"image {img.size} from {model} -> check_text2img.png{extra}"


def check_caption():
    out = nv.make_caption("fresh matcha latte at a cozy cafe", "Playful")
    if out["source"] != "llm":
        raise RuntimeError(f"fell back to template. Reason: {out.get('warning')}")
    return f"{out['caption']}  [{out['model']}]"


def check_img2img():
    ref = Image.new("RGB", (640, 640), (245, 235, 220))
    d = ImageDraw.Draw(ref)
    d.ellipse((160, 160, 480, 480), fill=(200, 60, 50))
    d.rectangle((300, 100, 340, 180), fill=(60, 120, 60))
    ref.save("check_reference.png")
    with open("check_reference.png", "rb") as f:
        raw = f.read()
    img, info = nv.image_plus_text_to_image("same apple on a festive diwali background, warm lighting", raw)
    img.save("check_img2img.png")
    extra = f" | notes: {info['warnings']}" if info["warnings"] else ""
    return (f"method={info['method']} reference_used={info['reference_used']} model={info['model']} "
            f"-> check_img2img.png | prompt sent: {info['final_prompt'][:140]}{extra}")


print("Creative Studio setup check")
print("Image order :", nv.IMAGE_MODELS, "+ Pollinations fallback" if nv.POLLINATIONS_FALLBACK else "")
print("Edit model  :", nv.EDIT_MODEL, f"(img2img mode: {nv.IMG2IMG_MODE})")
print("Groq key    :", "set" if nv.GROQ_API_KEY else "not set (optional)")

step("1. Keys + connection", check_keys)
if results and results[0][1]:
    step("2. Text -> image", check_text2img)
    step("3. Caption generator", check_caption)
    step("4. Image + text -> image", check_img2img)

print("\n==== SUMMARY ====")
for name, ok in results:
    print(("[PASS] " if ok else "[FAIL] ") + name)
if all(ok for _, ok in results) and len(results) == 4:
    print("\nAll good - start the server:  uvicorn main:app --host 0.0.0.0 --port 8000")
else:
    print("\nSomething failed - copy this whole output and send it over.")
