import os
import asyncio
import contextlib
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetDialogFiltersRequest

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────
load_dotenv()
API_ID = int(os.getenv("api_id"))
API_HASH = os.getenv("api_hash")
SESSION_NAME = "session_name"

TARGET_FILTER_ID = 5          # папка "Отбор" (DialogFilter id)
CONTEXT_COUNT = 5             # сколько сообщений истории брать
TYPING_EVERY_SEC = 3.0        # период "typing..." пока генерируем

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
target_chat_ids: set[int] = set()


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(chat_id: int, direction: str, text: str):
    """
    Пишем отдельный файл на каждое входящее/исходящее сообщение:
    logs/YYYY-MM-DD_HH-MM-SS_chat<id>_<IN|OUT>.log
    """
    ts_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"{ts_file}_chat{chat_id}_{direction}.log"
    (LOG_DIR / fn).write_text(
        f"{now_ts()} | {direction} | chat_id={chat_id}\n{text}\n",
        encoding="utf-8",
    )


def generate_reply(context_text: str) -> str:
    """
    Заглушка генерации. Сюда подключишь LLM/HTTP.
    """
    lines = [l for l in context_text.splitlines() if l.strip()]
    last = lines[-1] if lines else "..."
    return f"Автоответ:\nЯ увидел:\n{last}"


async def load_filter_chat_ids(filter_id: int) -> set[int]:
    """
    Берём DialogFilter по id и извлекаем include_peers -> entity.id
    """
    res = await client(GetDialogFiltersRequest())
    filters = res.filters if hasattr(res, "filters") else res

    f = next((x for x in filters if getattr(x, "id", None) == filter_id), None)
    if f is None:
        raise RuntimeError(f"Фильтр с id={filter_id} не найден")

    include_peers = getattr(f, "include_peers", None)
    if include_peers is None:
        raise RuntimeError(f"Фильтр id={filter_id} не содержит include_peers (возможно, системный)")

    ids = set()
    for p in include_peers:
        ent = await client.get_entity(p)
        ids.add(ent.id)
    return ids


async def keep_typing(chat_id: int, stop_event: asyncio.Event):
    """
    Пока не stop_event, периодически показываем 'typing...'
    """
    try:
        while not stop_event.is_set():
            async with client.action(chat_id, "typing"):
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=TYPING_EVERY_SEC)
                except asyncio.TimeoutError:
                    pass
    except Exception:
        return


@client.on(events.NewMessage)
async def on_new_message(event: events.NewMessage.Event):
    # не отвечаем на свои
    if event.out:
        return

    chat_id = event.chat_id
    if chat_id not in target_chat_ids:
        return

    # ─── запускать только если есть текст ─────────────────────────────
    incoming_text = (event.raw_text or "").strip()
    if not incoming_text:
        return

    # лог входящего
    log_line(chat_id, "IN", incoming_text)

    # ─── контекст ────────────────────────────────────────────────────
    history = await client.get_messages(chat_id, limit=CONTEXT_COUNT + 1)
    history = list(reversed(history))  # от старых к новым

    ctx_lines = []
    for m in history:
        txt = (m.message or "").strip()
        if not txt:
            continue
        sender = m.sender_id or "?"
        ctx_lines.append(f"{sender}: {txt}")
    context_text = "\n".join(ctx_lines)

    # ─── typing... пока генерируем ───────────────────────────────────
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(chat_id, stop_typing))

    try:
        reply_text = generate_reply(context_text)
    finally:
        stop_typing.set()
        with contextlib.suppress(Exception):
            await typing_task

    # ─── отправка и лог ──────────────────────────────────────────────
    await event.reply(reply_text)
    log_line(chat_id, "OUT", reply_text)

    chat = event.chat
    chat_name = getattr(chat, "title", None) or getattr(chat, "first_name", None) or str(chat_id)
    print(f"[INFO] IN/OUT записаны, ответ отправлен в: {chat_name} ({chat_id})")


async def main():
    await client.start()  # авторизация как пользователь
    me = await client.get_me()
    print(f"[INFO] Авторизован как: {me.id} {me.first_name} @{me.username}")

    global target_chat_ids
    target_chat_ids = await load_filter_chat_ids(TARGET_FILTER_ID)
    print(f"[INFO] Фильтр id={TARGET_FILTER_ID}: чатов = {len(target_chat_ids)}")
    print("[INFO] Слушаю новые текстовые сообщения...")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
