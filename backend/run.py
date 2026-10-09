"""Server launch script using Uvicorn."""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import uvicorn

# Ensure the backend directory is in the Python search path
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

if __name__ == "__main__":
    from app.config import settings

    print("=" * 65)
    print("🌟 AdVantage OS | Backend Server Launch")
    print(f"🔗 Swagger UI Docs : http://{settings.host}:{settings.port}/docs")
    print(f"🔗 ReDoc UI Docs   : http://{settings.host}:{settings.port}/redoc")
    print(f"🔗 Root Health     : http://{settings.host}:{settings.port}/health")
    print("=" * 65)

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
