"""FastAPI application entry point."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn

from app.api.routes import drowsiness_router, kid_safety_router, stress_router, visibility_router
from app.core.config import CORS_ORIGINS, DROWSINESS_SERVICE_DISABLED_REASON, ENABLE_DROWSINESS_SERVICE, HOST, PORT, TEST_MODE
from app.database.mongo import init_mongo
from app.services.drowsiness_service import start as start_drowsiness_service, stop as stop_drowsiness_service
from app.services.fog_service import load_model as load_fog_model
from app.services.kid_service import load_model as load_kid_model
from app.utils.logger import get_logger
from routes.auth import router as auth_router
from routes.api import router as api_router
from routes.analytics_routes import router as analytics_router
from routes.ws import router as ws_router

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
	init_mongo()
	if not TEST_MODE:
		load_fog_model()
		load_kid_model()
		if ENABLE_DROWSINESS_SERVICE:
			start_drowsiness_service()
		else:
			logger.warning("Drowsiness detection startup skipped: %s", DROWSINESS_SERVICE_DISABLED_REASON)
	else:
		logger.info("TEST_MODE enabled: heavy model startup skipped")

	logger.info("API ready on http://%s:%s", HOST, PORT)
	yield

	if not TEST_MODE and ENABLE_DROWSINESS_SERVICE:
		stop_drowsiness_service()


app = FastAPI(
	title="AI-Based Driver Safety Risk Prediction System",
	version="3.0.0",
	lifespan=lifespan,
)

port = int(os.environ.get("PORT", 10000))

app.add_middleware(
	CORSMiddleware,
	allow_origins=CORS_ORIGINS,
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
	return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	return JSONResponse(status_code=422, content={"error": "Validation failed", "details": exc.errors()})


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
	logger.exception("Unhandled error")
	return JSONResponse(status_code=500, content={"error": "Internal server error"})


@app.get("/health")
def health_check() -> dict:
	return {"status": "ok", "service": "driver-safety-api"}


@app.get("/api/status")
def status() -> dict:
	return {"status": "ok"}


app.include_router(drowsiness_router)
app.include_router(stress_router)
app.include_router(kid_safety_router)
app.include_router(visibility_router)
app.include_router(auth_router)
app.include_router(api_router)
app.include_router(analytics_router)
app.include_router(ws_router)


@app.get("/")
def root() -> dict:
	return {"message": "Driver safety API", "docs": "/docs"}


if __name__ == "__main__":
	uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
