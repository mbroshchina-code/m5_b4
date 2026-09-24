"""Общий вывод SSE-токенов в Telegram через sendMessageDraft."""

import logging
import uuid
from collections.abc import AsyncIterator

import httpx
from aiogram.enums import ChatAction
from aiogram.types import Message

from bot.services.backend_client import BackendStreamError
from bot.keyboards.inline import feedback_kb

log = logging.getLogger("bot")


def service_error_message(exc: Exception) -> str:
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "Сервис недоступен, попробуйте позже."
    if isinstance(exc, httpx.ReadTimeout):
        return "Ответ занимает слишком долго."
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        if code == 403:
            try:
                detail = exc.response.json().get("detail", {})
            except (ValueError, httpx.ResponseNotRead):
                detail = {}
            if detail.get("code") == "moderation_blocked":
                return "Не могу обработать этот запрос — он может нарушать правила."
        if code == 429:
            return "Слишком много запросов, подождите минуту."
        if code >= 500:
            return "Внутренняя ошибка сервиса."
    if isinstance(exc, BackendStreamError):
        return "Внутренняя ошибка сервиса."
    return "Не удалось получить ответ. Попробуйте переформулировать вопрос."


async def stream_to_chat(message: Message, tokens: AsyncIterator[str]) -> str:
    """Один draft_id на поток; обычное сообщение фиксирует результат."""
    draft_id = uuid.uuid4().int & 0xFFFFFFFF
    buffer = ""
    error_message: str | None = None

    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING,
    )

    try:
        async for delta in tokens:
            buffer += delta
            if not buffer.strip():
                continue
            # Telegram принимает не более 4096 символов в одном сообщении.
            display = buffer if len(buffer) <= 4096 else buffer[:4093] + "..."
            await message.bot.send_message_draft(
                chat_id=message.chat.id,
                text=display,
                draft_id=draft_id,
            )
    except Exception as exc:
        log.warning("backend stream failed: %s", exc, exc_info=True)
        error_message = service_error_message(exc)

    replacement = getattr(tokens, "replacement", None)
    if replacement:
        buffer = replacement
        await message.bot.send_message_draft(
            chat_id=message.chat.id,
            text=buffer,
            draft_id=draft_id,
        )

    if not buffer.strip():
        buffer = error_message or "Не удалось получить ответ. Попробуйте позже."

    # Ответы этого ассистента ограничены max_tokens и обычно короче лимита.
    # Защитный срез не позволяет Telegram отклонить финальную отправку.
    final_text = buffer if len(buffer) <= 4096 else buffer[:4093] + "..."
    message_id = getattr(tokens, "message_id", None)
    reply_markup = feedback_kb(str(message_id)) if message_id else None
    await message.bot.send_message(
        chat_id=message.chat.id,
        text=final_text,
        reply_markup=reply_markup,
    )
    return buffer
