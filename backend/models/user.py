"""Pydantic models for authentication and user payloads."""

from datetime import datetime

from pydantic import BaseModel, Field

from models.types import EmailAddress


class UserCreate(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: EmailAddress
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailAddress
    password: str = Field(min_length=8, max_length=128)


class UserPublic(BaseModel):
    id: str
    name: str
    email: EmailAddress
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic
