"""Authentication routes: register, login, forgot/reset password via OTP."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from database.mongo import get_user_by_email, update_user_password
from models.types import EmailAddress
from models.user import TokenResponse, UserCreate, UserLogin
from services.auth_service import login_user, register_user
from services.otp_service import consume_otp, request_otp, verify_otp
from utils.password_hash import hash_password

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Pydantic models for OTP flow ──────────────────────────────────────

class ForgotPasswordRequest(BaseModel):
    email: EmailAddress


class VerifyOTPRequest(BaseModel):
    email: EmailAddress
    otp_code: str = Field(min_length=6, max_length=6)


class ResetPasswordRequest(BaseModel):
    email: EmailAddress
    otp_code: str = Field(min_length=6, max_length=6)
    new_password: str = Field(min_length=8, max_length=128)


# ── Routes ────────────────────────────────────────────────────────────

@router.post("/register")
def register(payload: UserCreate):
    try:
        user = register_user(payload)
        return {"message": "User registered successfully", "user": user.model_dump()}
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise exc
        return {"error": str(exc)}


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLogin):
    try:
        token, user = login_user(payload.email, payload.password)
        return TokenResponse(access_token=token, user=user)
    except Exception as exc:
        if hasattr(exc, "status_code"):
            raise exc
        return {"error": str(exc)}


@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordRequest):
    """
    Trigger OTP generation for password reset.

    Always returns 200 to avoid leaking whether the email exists.
    The OTP is emailed (or logged to console in dev mode).
    """
    try:
        # Only send OTP if user exists — but never reveal non-existence to caller
        user = get_user_by_email(payload.email)
        if user:
            result = request_otp(payload.email)
            # In dev mode (no SMTP), expose the OTP in the response for easy testing
            if result.get("dev_otp"):
                return {
                    "message": "OTP generated (SMTP not configured — dev mode)",
                    "dev_otp": result["dev_otp"],
                }
        return {"message": "If that email is registered, an OTP has been sent."}
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/verify-otp")
def verify_otp_endpoint(payload: VerifyOTPRequest):
    """Verify that the submitted OTP is valid and unexpired."""
    try:
        valid = verify_otp(payload.email, payload.otp_code)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP",
            )
        return {"message": "OTP verified successfully", "valid": True}
    except HTTPException:
        raise
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest):
    """Reset the user password after OTP verification."""
    try:
        valid = verify_otp(payload.email, payload.otp_code)
        if not valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired OTP",
            )

        updated = update_user_password(payload.email, hash_password(payload.new_password))
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found or database unavailable",
            )

        consume_otp(payload.email)
        return {"message": "Password reset successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        return {"error": str(exc)}
