"""Доменные модели чата.

Чистые Pydantic v2-модели — никаких импортов из SQLAlchemy / FastAPI / aiofiles.
Граница между доменом и инфраструктурой проходит через ORM-границу
(`ChatMessage.model_validate(row, from_attributes=True)`).
"""
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["user", "assistant", "system"]


class ChatMessage(BaseModel):
    """Сообщение внутри чата (доменная модель)."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    chat_id: UUID
    role: Role
    content: str
    media_refs: dict | None = None
    tokens: int | None = None
    latency_ms: float | None = None
    prompt_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chat(BaseModel):
    """Чат — один диалог одного пользователя с ассистентом."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(default_factory=uuid4)
    owner_external_id: str
    interface: str
    system_prompt: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SystemPrompt(BaseModel):
    """Версионированный системный промпт — кандидат A/B traffic-split.

    Активные кандидаты выбираются репозиторием (active=TRUE и traffic_pct>0),
    конкретный вариант для пользователя — `prompt_selection.choose_by_split`.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: str
    body: str
    traffic_pct: int
