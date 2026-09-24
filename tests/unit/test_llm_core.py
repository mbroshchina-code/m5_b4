"""Юнит-тесты актуального LLMService без сетевых вызовов."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.schemas.chat import ChatRequest
from app.services.llm import LLMService


def _request(temperature: float = 0.2) -> ChatRequest:
    return ChatRequest(
        messages=[{"role": "user", "content": "ошибка оплаты"}],
        temperature=temperature,
    )


def test_chat_request_defaults_to_configured_model():
    assert _request().model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_stream_uses_single_required_openai_call():
    async def chunks():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Баг"))],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        )

    create = AsyncMock(return_value=chunks())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    service = LLMService(client, cache=None)

    result = [delta async for delta in service.stream(_request())]

    create.assert_awaited_once()
    assert create.await_args.kwargs == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "ошибка оплаты"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    assert result[0].content == "Баг"
    assert result[1].usage.total_tokens == 12


@pytest.mark.asyncio
async def test_complete_returns_cached_deterministic_response():
    cached = (
        '{"content":"Из кеша","model":"gpt-4o-mini",'
        '"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},'
        '"finish_reason":"stop","cached":false}'
    )
    cache = SimpleNamespace(get=AsyncMock(return_value=cached), setex=AsyncMock())
    service = LLMService(llm=None, cache=cache)

    response = await service.complete(_request(temperature=0.0))

    assert response.content == "Из кеша"
    assert response.cached is True
    cache.setex.assert_not_awaited()


def test_cache_key_does_not_expose_prompt():
    service = LLMService(llm=None, cache=None)
    key = service._key(_request())
    assert key.startswith("chat:")
    assert "ошибка" not in key
