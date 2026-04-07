# server/app.py — OpenEnv entry point
# Re-exports the FastAPI app from server.py at the root level
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import app

__all__ = ["app"]
