"""
AdVantage OS — Backend Package.
Modular AI-driven backend for SMEs.
"""

from pathlib import Path
import sys

_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

try:
    from backend.main import app
except ImportError:
    try:
        from main import app
    except ImportError:
        app = None

__all__ = ["app"]
