"""Authentication service for user registration/login and auth dependencies."""

from datetime import datetime

from fastapi import Depends, HTTPException, status

from database.mongo import create_user, get_user_by_email, get_user_by_id
from models.user import UserCreate, UserPublic
from utils.jwt_handler import create_access_token, decode_access_token, get_bearer_token
from utils.password_hash import hash_password, verify_password


def _to_public_user(user_record: dict) -> UserPublic:
    created_at = user_record.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    return UserPublic(
        id=user_record["id"],
        name=user_record["name"],
        email=user_record["email"],
        created_at=created_at,
    )


def register_user(payload: UserCreate) -> UserPublic:
    existing = get_user_by_email(payload.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    user = create_user(
        name=payload.name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    if not user:
        raise HTTPException(status_code=503, detail="Database unavailable")

    return _to_public_user(user)


def login_user(email: str, password: str) -> tuple[str, UserPublic]:
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(password, user.get("hashed_password", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(subject=user["id"], extra_claims={"email": user["email"]})
    return token, _to_public_user(user)


def get_current_user(token: str = Depends(get_bearer_token)) -> dict:
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return user
