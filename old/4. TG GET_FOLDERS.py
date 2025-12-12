import asyncio
import os
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.functions.messages import GetDialogFiltersRequest

load_dotenv()
API_ID = int(os.getenv("api_id"))
API_HASH = os.getenv("api_hash")
SESSION_NAME = "session_name"

client = TelegramClient(SESSION_NAME, API_ID, API_HASH)


async def resolve_peer(peer):
    """Вернуть человекочитаемое имя чата/пользователя по InputPeer."""
    try:
        entity = await client.get_entity(peer)

        # Канал / супергруппа / группа
        if hasattr(entity, "title"):
            return f"[GROUP] {entity.title} (id={entity.id})"

        # Пользователь
        if hasattr(entity, "first_name"):
            ln = f" {entity.last_name}" if getattr(entity, "last_name", None) else ""
            return f"[USER] {entity.first_name}{ln} (id={entity.id})"

        return f"[UNKNOWN ENTITY] id={getattr(entity, 'id', '?')}"
    except Exception as e:
        return f"[FAILED RESOLVE] {peer} ({e})"


async def main():
    await client.start()
    me = await client.get_me()
    print(f"[INFO] Авторизован как: {me.id} {me.first_name} @{me.username}\n")

    res = await client(GetDialogFiltersRequest())
    filters = res.filters if hasattr(res, "filters") else res

    if not filters:
        print("Нет папок.")
        return

    print("\n=== НАЙДЕННЫЕ ПАПКИ ===\n")

    for f in filters:
        print("─────────────────────────────────────────────")
        print(f"Тип: {type(f).__name__}")
        print(f)
        # Название, если есть
        title_obj = getattr(f, "title", None)
        title_text = getattr(title_obj, "text", None) if title_obj is not None else None
        if title_text:
            print(f"Папка: {title_text}")
        else:
            print("Папка: (без названия)")

        # Для DialogFilterDefault и других "особых" конструкторов полей может не быть
        include_peers = getattr(f, "include_peers", None)

        if include_peers is None:
            # Это как раз DialogFilterDefault (по доке у него нет членов)
            print("  У этого фильтра нет списка include_peers (скорее всего, системная папка по умолчанию).")
            print()
            continue

        print(f"Всего include_peers: {len(include_peers)}")

        if not include_peers:
            print("  (папка пустая)\n")
            continue

        print("  Чаты внутри папки:")
        for peer in include_peers:
            resolved = await resolve_peer(peer)
            print("   •", resolved)
        print()


if __name__ == "__main__":
    asyncio.run(main())
