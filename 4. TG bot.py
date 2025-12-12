import os
import gc
import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import psutil
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetDialogFiltersRequest

# ───────────────────────────── CONFIG ─────────────────────────────
load_dotenv()

API_ID = int(os.getenv("api_id"))
API_HASH = os.getenv("api_hash")
SESSION_NAME = os.getenv("tg_session_name") or "session_name"

TARGET_FILTER_ID = int(os.getenv("tg_filter_id") or "5")  # "Отбор" = 5
CONTEXT_COUNT = int(os.getenv("tg_context_count") or "5")  # не обязательно, есть своя история
TYPING_EVERY_SEC = float(os.getenv("tg_typing_every_sec") or "4.0")

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL_ID = os.getenv("base_model_id") or "mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM_PROMPT = os.getenv("system_prompt") or (
    "Ты дружелюбный и лаконичный помощник.\n"
    "Главный фокус — переписки и вопросы по IT: отвечай по делу, без лишней воды."
)
USER_INSTRUCTION_TEMPLATE = "Имя собеседника: {who}. Напиши ответ на сообщение: {text}"

MAX_CONTEXT_TOKENS = int(os.getenv("max_context_tokens") or "2048")
MAX_HISTORY_MESSAGES = int(os.getenv("max_history_messages") or "40")

ADMIN_ID = int(os.getenv("admin_id") or "304622290")  # кто может менять параметры

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


@dataclass
class RuntimeParams:
    lora_adapter_dir: str = "models/vlad/final_adapter"
    max_new_tokens: int = 128
    temperature: float = 0.5
    top_p: float = 0.8

    whoo_default: str = "Без имени"   # дефолт, если имени в чате нет
    whoo_locked: bool = False         # если True — используем whoo_value
    whoo_value: str = "Без имени"     # значение, когда locked=True


P = RuntimeParams(
    lora_adapter_dir=os.getenv("lora_adapter_dir") or "models/vlad/final_adapter",
    max_new_tokens=int(os.getenv("max_new_tokens") or "128"),
    temperature=float(os.getenv("temperature") or "0.5"),
    top_p=float(os.getenv("top_p") or "0.8"),
    whoo_default=os.getenv("whoo_default") or "Без имени",
    whoo_locked=(os.getenv("whoo_locked") or "false").lower() == "true",
    whoo_value=os.getenv("whoo_value") or "Без имени",
)

# ───────────────────────────── LOGGING ─────────────────────────────
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_msg(chat_id: int, direction: str, text: str) -> None:
    ts_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fn = f"{ts_file}_chat{chat_id}_{direction}.log"
    (LOG_DIR / fn).write_text(
        f"{now_ts()} | {direction} | chat_id={chat_id}\n{text}\n",
        encoding="utf-8",
    )


def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


# ───────────────────────────── LLM ─────────────────────────────
tokenizer = None
base_model = None
model = None

DIALOGS: Dict[int, List[dict]] = {}  # chat_id -> [{"role","content"}, ...]


def reset_dialog(chat_id: int) -> None:
    DIALOGS[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]


def trim_history(chat_id: int) -> List[dict]:
    history = DIALOGS.get(chat_id, [])
    if not history:
        return []
    head = history[:1] if history and history[0].get("role") == "system" else []
    tail = history[1:]
    if len(tail) > MAX_HISTORY_MESSAGES:
        tail = tail[-MAX_HISTORY_MESSAGES:]
    DIALOGS[chat_id] = head + tail
    return DIALOGS[chat_id]


def build_chat_messages(messages: List[dict], who: str) -> List[dict]:
    templated = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            content = USER_INSTRUCTION_TEMPLATE.format(who=who, text=content)
        templated.append({"role": role, "content": content})

    merged = []
    for m in templated:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)

    has_system = merged and merged[0]["role"] == "system"
    core = merged[1:] if has_system else merged

    while core and core[0]["role"] == "assistant":
        core = core[1:]

    alternated = [{"role": "system", "content": merged[0]["content"]}] if has_system else []
    for m in core:
        if not alternated:
            if m["role"] == "assistant":
                continue
            alternated.append(m)
            continue
        if alternated[-1]["role"] == m["role"]:
            alternated[-1]["content"] += "\n" + m["content"]
        else:
            alternated.append(m)

    while alternated and alternated[-1]["role"] != "user":
        alternated.pop()

    return alternated


def current_gen_cfg() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=P.max_new_tokens,
        temperature=P.temperature,
        top_p=P.top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )


def load_llm(lora_dir: str) -> None:
    global tokenizer, base_model, model

    # выгружаем старое
    with contextlib.suppress(Exception):
        if model is not None:
            del model
        if base_model is not None:
            del base_model
        if tokenizer is not None:
            del tokenizer
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"[LLM] Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[LLM] Loading base model: {BASE_MODEL_ID} ({DEVICE}/{DTYPE})")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"[LLM] Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_dir,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )
    model.eval()
    print("[LLM] Ready.")


@torch.inference_mode()
def llm_answer(chat_id: int, text: str, who: str) -> str:
    if chat_id not in DIALOGS:
        reset_dialog(chat_id)

    DIALOGS[chat_id].append({"role": "user", "content": text})
    history = trim_history(chat_id)
    chat_ctx = build_chat_messages(history, who)
    if not chat_ctx:
        return "История пуста, отправьте сообщение ещё раз."

    prompt_text = tokenizer.apply_chat_template(
        chat_ctx,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(prompt_ids) > MAX_CONTEXT_TOKENS:
        prompt_ids = prompt_ids[-MAX_CONTEXT_TOKENS:]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    inputs = {
        "input_ids": torch.tensor([prompt_ids], device=model.device),
        "attention_mask": torch.ones(1, len(prompt_ids), device=model.device),
    }

    output_ids = model.generate(**inputs, generation_config=current_gen_cfg())[0]
    answer_ids = output_ids[len(prompt_ids):]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    DIALOGS[chat_id].append({"role": "assistant", "content": answer})
    trim_history(chat_id)
    return answer


# ───────────────────────────── TELEGRAM (USER) ─────────────────────────────
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
target_chat_ids: set[int] = set()


async def keep_typing(chat_id: int, stop_event: asyncio.Event) -> None:
    try:
        while not stop_event.is_set():
            async with client.action(chat_id, "typing"):
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=TYPING_EVERY_SEC)
                except asyncio.TimeoutError:
                    pass
    except Exception:
        return


async def load_filter_chat_ids(filter_id: int) -> set[int]:
    res = await client(GetDialogFiltersRequest())
    filters = res.filters if hasattr(res, "filters") else res

    f = next((x for x in filters if getattr(x, "id", None) == filter_id), None)
    if f is None:
        raise RuntimeError(f"Фильтр с id={filter_id} не найден")

    include_peers = getattr(f, "include_peers", None)
    if include_peers is None:
        raise RuntimeError(f"Фильтр id={filter_id} не содержит include_peers (похоже на системный)")

    ids = set()
    for p in include_peers:
        ent = await client.get_entity(p)
        ids.add(ent.id)
    return ids


def get_sender_name(event: events.NewMessage.Event) -> str:
    """
    WHOO — имя собеседника.
    Если WHOO закреплён (/who Имя) — используем его.
    Иначе берём из Telegram, а если пусто — whoo_default ("Без имени").
    """
    if P.whoo_locked and (P.whoo_value or "").strip():
        return P.whoo_value.strip()

    s = event.sender
    # sender может быть None в некоторых случаях — используем chat/дефолт
    name_parts = []
    if s is not None:
        fn = (getattr(s, "first_name", None) or "").strip()
        ln = (getattr(s, "last_name", None) or "").strip()
        un = (getattr(s, "username", None) or "").strip()
        if fn:
            name_parts.append(fn)
        if ln:
            name_parts.append(ln)
        if not name_parts and un:
            name_parts.append(un)

    name = " ".join(name_parts).strip()
    return name or P.whoo_default


def is_admin(event: events.NewMessage.Event) -> bool:
    s = event.sender
    return bool(s and getattr(s, "id", None) == ADMIN_ID)


async def reply_safe(event: events.NewMessage.Event, text: str) -> None:
    # Телега режет длинные сообщения — отправим кусками
    chunk_size = 4000
    for i in range(0, len(text), chunk_size):
        await event.reply(text[i:i + chunk_size])


def params_text() -> str:
    return (
        f"DEVICE={DEVICE}\n"
        f"RAM={ram_mb():.0f} MB\n\n"
        f"LORA_ADAPTER_DIR={P.lora_adapter_dir}\n"
        f"MAX_NEW_TOKENS={P.max_new_tokens}\n"
        f"TEMPERATURE={P.temperature}\n"
        f"TOP_P={P.top_p}\n\n"
        f"WHOO_LOCKED={P.whoo_locked}\n"
        f"WHOO_VALUE={P.whoo_value}\n"
        f"WHOO_DEFAULT={P.whoo_default}"
    )


async def handle_command(event: events.NewMessage.Event, text: str) -> bool:
    """
    Возвращает True, если сообщение было командой и обработано.
    Команды доступны только ADMIN_ID (кроме /params, /help для диагностики).
    """
    t = text.strip()
    if not t.startswith("/"):
        return False

    cmd, *args = t.split()

    if cmd in ("/help", "/start"):
        await reply_safe(
            event,
            "Команды:\n"
            "/params — показать параметры\n"
            "/set <param> <value> — изменить (admin)\n"
            "  param: max_new_tokens | temperature | top_p\n"
            "/who <Имя> — закрепить WHOO (admin)\n"
            "/who — снять закрепление WHOO (admin)\n"
            "/reload_lora <path> — перезагрузить LoRA (admin)\n"
            "/clear — очистить историю этого чата (admin)\n",
        )
        return True

    if cmd == "/params":
        await reply_safe(event, params_text())
        return True

    # дальше — только админ
    if not is_admin(event):
        await reply_safe(event, "Недостаточно прав.")
        return True

    if cmd == "/clear":
        reset_dialog(event.chat_id)
        await reply_safe(event, "Контекст очищен для этого чата.")
        return True

    if cmd == "/who":
        if args:
            P.whoo_value = " ".join(args).strip()
            P.whoo_locked = True
            await reply_safe(event, f"WHOO закреплён: {P.whoo_value}")
        else:
            P.whoo_locked = False
            await reply_safe(event, "WHOO больше не закреплён — имя будет браться из чата (или 'Без имени').")
        return True

    if cmd == "/set":
        if len(args) < 2:
            await reply_safe(event, "Формат: /set <param> <value>")
            return True
        param = args[0].lower()
        value = " ".join(args[1:]).strip()

        try:
            if param == "max_new_tokens":
                P.max_new_tokens = int(value)
            elif param == "temperature":
                P.temperature = float(value)
            elif param == "top_p":
                P.top_p = float(value)
            else:
                await reply_safe(event, "Неизвестный параметр. Доступно: max_new_tokens, temperature, top_p")
                return True
        except Exception as e:
            await reply_safe(event, f"Ошибка значения: {e}")
            return True

        await reply_safe(event, "Ок.\n" + params_text())
        return True

    if cmd == "/reload_lora":
        if not args:
            await reply_safe(event, "Формат: /reload_lora <path>")
            return True
        new_path = " ".join(args).strip()
        try:
            load_llm(new_path)
            P.lora_adapter_dir = new_path
            await reply_safe(event, f"LoRA перезагружен: {new_path}")
        except Exception as e:
            await reply_safe(event, f"Ошибка reload_lora: {e}")
        return True

    return False


@client.on(events.NewMessage)
async def on_new_message(event: events.NewMessage.Event):
    # не реагируем на свои исходящие
    if event.out:
        return

    chat_id = event.chat_id
    if chat_id not in target_chat_ids:
        return

    incoming_text = (event.raw_text or "").strip()
    if not incoming_text:
        return  # только текст

    # команды обрабатываем отдельно (и не запускаем генерацию)
    if await handle_command(event, incoming_text):
        return

    who = get_sender_name(event)

    # лог входящего
    log_msg(chat_id, "IN", f"WHO={who}\n{incoming_text}")

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(chat_id, stop_typing))

    try:
        answer = llm_answer(chat_id, incoming_text, who)
    finally:
        stop_typing.set()
        with contextlib.suppress(Exception):
            await typing_task

    # лог исходящего + отправка
    log_msg(chat_id, "OUT", f"WHO={who}\n{answer}")
    await reply_safe(event, answer)


async def main():
    # 1) грузим модель
    load_llm(P.lora_adapter_dir)

    # 2) логинимся в телеграм как пользователь
    await client.start()
    me = await client.get_me()
    print(f"[INFO] Telegram user: {me.id} {me.first_name} @{me.username}")

    # 3) берём чаты из папки id=5
    global target_chat_ids
    target_chat_ids = await load_filter_chat_ids(TARGET_FILTER_ID)
    print(f"[INFO] Filter id={TARGET_FILTER_ID}: chats={len(target_chat_ids)}")
    print("[INFO] Listening...")

    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
