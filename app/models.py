from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class SessionCreateRequest(BaseModel):
    provider: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(
        default=None,
        min_length=8,
        max_length=1000,
    )
    project: str = Field(
        default="portfolio",
        min_length=1,
        max_length=80,
    )
    ttl_seconds: int | None = Field(
        default=None,
        ge=60,
        le=3600,
    )

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        value = value.strip().lower()

        if not value:
            raise ValueError("Provider cannot be empty")

        return value

    @field_validator("model", "project")
    @classmethod
    def strip_values(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Value cannot be empty")

        return value


class SessionCreateResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    provider: str
    model: str
    project: str


class SessionHandoffRequest(BaseModel):
    project: str = Field(min_length=1, max_length=80)

    @field_validator("project")
    @classmethod
    def normalize_project(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("Project cannot be empty")
        return value


class SessionHandoffResponse(BaseModel):
    code: str
    expires_in: int
    project: str


class SessionExchangeRequest(BaseModel):
    code: str = Field(min_length=32, max_length=128)
    project: str = Field(min_length=1, max_length=80)

    @field_validator("code")
    @classmethod
    def strip_code(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Code cannot be empty")
        return value

    @field_validator("project")
    @classmethod
    def normalize_project(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("Project cannot be empty")
        return value

class SessionExchangeResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    provider: str
    model: str
    target_project: str


class ProviderModel(BaseModel):
    id: str
    provider: str
    model_id: str
    label: str | None = None
    recommended: bool = False


class ProvidersResponse(BaseModel):
    providers: list[ProviderModel]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None

    messages: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=256,
    )

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
    )

    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=32768,
    )

    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        le=32768,
    )

    stream: bool = False

    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None

    stop: str | list[str] | None = None
    seed: int | None = None
    user: str | None = None


class RevokeResponse(BaseModel):
    revoked: bool = True