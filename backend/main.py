"""Compatibility entrypoint that exposes the unified FastAPI app."""

import os

import uvicorn

from app.main import app


port = int(os.environ.get("PORT", 10000))


if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
