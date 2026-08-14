from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SessionCreateRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=100)
    api_key: str | None = Field(default=None, min_length=8, max_length=1000)
    project: str = Field(default="portfolio", min_length=1, max_length=80)
    ttl_seconds: int | None = Field(default=None, ge=300, le=3600)

    @field_validator("provider", "model", "project")
    @classmethod
    def strip_values(cls, value: str) -> str:
        return value.strip()


class SessionCreateResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    provider: str
    model: str
    project: str


class ProviderModel(BaseModel):
    id: str
    provider: str
    label: str | None = None
    recommended: bool = False


class ProvidersResponse(BaseModel):
    providers: list[ProviderModel]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=16384)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=16384)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None


class RevokeResponse(BaseModel):
    revoked: bool = True
