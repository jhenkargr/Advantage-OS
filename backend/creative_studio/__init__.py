"""
creative_studio
===============
AdVantage OS — AI-driven Creative Generation Studio.

Exposes:
  - router: APIRouter with endpoints:
      • /image_generation/health  and  /generate/health
      • /image_generation/text2img and  /generate/text2img
      • /image_generation/img2img  and  /generate/img2img
      • /image_generation/caption  and  /generate/caption
  - nvidia_client: Multi-provider backend (NVIDIA API Catalog + Groq + Pollinations)
"""

from .creative import router, EXPOSED_HEADERS, app
from . import nvidia_client

__all__ = ["router", "EXPOSED_HEADERS", "nvidia_client", "app"]
