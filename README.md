# Creative Studio - NVIDIA + Groq + Pollinations (free providers, automatic fallbacks)

Three independent tools:

| Endpoint | Input | Output |
|---|---|---|
| `POST /generate/text2img` | prompt | PNG image |
| `POST /generate/img2img` | prompt + reference image | PNG image |
| `POST /generate/caption` | description + tone | JSON caption |

## Who answers what (and the fallbacks)
| Job | Order tried |
|---|---|
| Text -> image | NVIDIA FLUX.1-dev -> NVIDIA FLUX.1-schnell -> Pollinations (free backup) |
| Captions | Groq Llama (free key, recommended) -> NVIDIA chat models -> a template |
| Image + text -> image | NVIDIA Kontext if it accepts your image, else a vision model describes it and the image chain above generates |

Every response tells you who answered (`X-Model-Used` header for images, `model` field for captions).
NVIDIA's free tier is shared and sometimes slow or errors; the fallbacks exist so one flaky provider does not
break your app. A safety-filter block (HTTP 422) is never retried on another provider.

## 1. Get your free keys
1. **NVIDIA (required):** https://build.nvidia.com/settings/api-keys -> Generate Key (starts with `nvapi-`).
2. **Groq (recommended, for captions):** sign up at https://console.groq.com (no credit card), open
   https://console.groq.com/keys -> Create API Key (starts with `gsk_`). Free tier is about 30 requests/minute.
3. **Pollinations (optional):** https://enter.pollinations.ai -> create a free key. Without it the backup image
   provider still works but is throttled (about 1 request per 15 s) and may add a watermark.

Keep keys in `.env` only. Never paste them in chat or upload `.env` anywhere.

## 2. Set up (VS Code terminal, PowerShell)
```
venv\Scripts\activate
pip install -r requirements.txt
code .env
```
`.env` should look like this (use your real keys):
```
NVIDIA_API_KEY=nvapi-...
GROQ_API_KEY=gsk_...
POLLINATIONS_API_KEY=
```
(`.env.example` shows every option.)

## 3. Self-test
```
python check_setup.py
```
It tests the real pipeline the way the server uses it (with fallbacks) and prints which provider answered each
step. Expect four `[PASS]`. Images can take 10 to 150 seconds on a busy free tier. Send the whole output if anything
fails (remove any key if one is ever printed).

## 4. Run the server
```
uvicorn main:app --host 0.0.0.0 --port 8000
```
Test everything at **http://127.0.0.1:8000/docs** (forms and upload buttons).
`http://127.0.0.1:8000/health` shows the provider order in use and any models temporarily skipped.

PowerShell curl equivalents (note `curl.exe`):
```
curl.exe -X POST http://127.0.0.1:8000/generate/text2img -F "prompt=professional photo of a matcha latte in a cozy cafe, warm light" -o out.png
curl.exe -X POST http://127.0.0.1:8000/generate/img2img -F "prompt=same product on a festive diwali background" -F "reference_image=@photo.jpg" -o out2.png
curl.exe -X POST http://127.0.0.1:8000/generate/caption -F "description=matcha latte at a cozy cafe" -F "tone=Playful"
```

## Image + text -> image: what it really does
NVIDIA's hosted Kontext accepts only a few built-in sample images, not your uploads (it returned 422 in your
test, as documented). So the backend uses **describe-and-generate**: a vision model reads your image and writes one
prompt that keeps its subject, colors and style plus your requested change, then FLUX generates from it. It keeps
the idea and look, not the exact pixels or layout. `X-Img2Img-Method` and `X-Final-Prompt` headers show what happened.

## Notes for your report
- NVIDIA retires models regularly (Llama 3.1 8B and 3.3 70B ended 2026-08-26) and its model list can include models
  that are not callable for your account; the backend reads the live list and falls through automatically.
- FLUX.1-dev and Kontext-dev are non-commercial licenses; free tiers are trial/shared services. Fine for a college
  project; say so if asked about production use.

## Troubleshooting
| Symptom | Meaning / fix |
|---|---|
| `NVIDIA rejected the API key (401/403)` | Wrong/deleted NVIDIA key. Make a new one, fix `.env`. |
| Groq key rejected | Groq is skipped automatically and NVIDIA answers; fix the `gsk_` key when convenient. |
| Slow images | Shared free tier. After a slow/failed model the next requests skip it for 5 minutes. |
| `All image providers failed` | NVIDIA and Pollinations both failed at that moment. Retry in a minute. |
| `429` | Per-minute limit. Wait about a minute. |
| `422 safety filter` | The prompt/output was blocked. Rephrase. |

## Files
`nvidia_client.py` (all provider calls, reusable by other modules), `main.py` (endpoints), `check_setup.py`
(self-test), `.env.example`, `requirements.txt`.
