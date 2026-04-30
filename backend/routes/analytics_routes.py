"""Analytics API routes."""

from fastapi import APIRouter

from services.analytics_service import generate_summary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def get_analytics_summary() -> dict:
    return generate_summary(user_id="system")
