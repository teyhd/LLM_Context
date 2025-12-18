# tg_llm_userbot.py
# Требования: pip install telethon python-dotenv transformers peft torch psutil
# (и CUDA/torch по желанию)
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
# ───────────────────────────── ENV/CONFIG ─────────────────────────────
load_dotenv()
API_ID = int(os.getenv("api_id"))
API_HASH = os.getenv("api_hash")
SESSION_NAME = os.getenv("tg_session_name") or "session_name"
TARGET_FILTER_ID = int(os.getenv("tg_filter_id") or "5")  # "Отбор" = 5
TYPING_EVERY_SEC = float(os.getenv("tg_typing_every_sec") or "4.0")
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
BASE_MODEL_ID = os.getenv("base_model_id") or "mistralai/Mistral-7B-Instruct-v0.3"
SYSTEM_PROMPT = os.getenv("system_prompt") or "Ты Влад. Ты дружелюбный и лаконичный.\nГлавный фокус — переписка: отвечай по делу, без лишней воды."
USER_INSTRUCTION_TEMPLATE = "Имя собеседника: {who}. Напиши ответ на сообщение: {text}"
MAX_CONTEXT_TOKENS = int(os.getenv("max_context_tokens") or "2048")
MAX_HISTORY_MESSAGES = int(os.getenv("max_history_messages") or "40")
HISTORY_BACKFILL_MESSAGES = int(os.getenv("history_backfill_messages") or "10")
MIN_CONFIDENCE = float(os.getenv("min_confidence") or "0.25")
ADMIN_ID = int(os.getenv("admin_id") or "304622290")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
COUNT = 0
STOPED = False
@dataclass
class RuntimeParams:
    lora_adapter_dir: str = "models/vlad/final_adapter"
    max_new_tokens: int = 128
    temperature: float = 0.5
    top_p: float = 0.8
    top_k: float = 40
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    whoo_default: str = "Без имени"
    whoo_locked: bool = False
    whoo_value: str = "Без имени"
P = RuntimeParams(
    lora_adapter_dir=os.getenv("lora_adapter_dir") or "models/vlad/final_adapter",
    max_new_tokens=int(os.getenv("max_new_tokens") or "128"),
    temperature=float(os.getenv("temperature") or "0.2"),
    top_p=float(os.getenv("top_p") or "0.8"),
    top_k=int(os.getenv("top_k") or "40"),
    repetition_penalty=float(os.getenv("repetition_penalty") or "1.0"),
    no_repeat_ngram_size=int(os.getenv("no_repeat_ngram_size") or "4"),
    whoo_default=os.getenv("whoo_default") or "Без имени",
    whoo_locked=(os.getenv("whoo_locked") or "false").lower() == "true",
    whoo_value=os.getenv("whoo_value") or "Без имени",
)
# ───────────────────────────── HELPERS ─────────────────────────────
def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20
def _safe_name(s: str) -> str:
    s = (s or "").strip()
    for ch in '<>:"/\\|?*\n\r\t':
        s = s.replace(ch, "_")
    s = " ".join(s.split())
    return (s[:80] or "noname")
def _log_path(chat_id: int, sender_id: int, who: str) -> Path:
    chat_dir = LOG_DIR / f"chat_{chat_id}"
    chat_dir.mkdir(parents=True, exist_ok=True)
    return chat_dir / f"user_{sender_id}_{_safe_name(who)}.log"
def log_msg(chat_id: int, sender_id: int, who: str, direction: str, text: str) -> None:
    """
    Один файл на пользователя (в рамках чата). Каждая строка с временем.
    """
    path = _log_path(chat_id, sender_id, who)
    safe_text = (text or "").replace("\n", "\\n")
    line = f"{now_ts()} | {direction:<3} | {safe_text}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
def params_text() -> str:
    return (
        f"DEVICE={DEVICE}\n"
        f"RAM={ram_mb():.0f} MB\n\n"
        f"LORA_ADAPTER_DIR={P.lora_adapter_dir}\n"
        f"MAX_NEW_TOKENS={P.max_new_tokens}\n"
        f"TEMPERATURE={P.temperature}\n"
        f"TOP_P={P.top_p}\n"
        f"HISTORY_BACKFILL_MESSAGES={HISTORY_BACKFILL_MESSAGES}\n\n"
        f"MIN_CONFIDENCE={MIN_CONFIDENCE}\n"
        f"WHOO_LOCKED={P.whoo_locked}\n"
        f"WHOO_VALUE={P.whoo_value}\n"
        f"WHOO_DEFAULT={P.whoo_default}"
    )
async def reply_safe(event: events.NewMessage.Event, text: str) -> None:
    chunk_size = 4000
    if text == False:
        return
    for i in range(0, len(text), chunk_size):
        await event.respond(text[i:i + chunk_size])
       # await client.send_message(await get_whoo(event),text[i:i + chunk_size])
# ───────────────────────────── LLM LOAD/GEN ─────────────────────────────
tokenizer = None
base_model = None
model = None
DIALOGS: Dict[int, List[dict]] = {}  # chat_id -> history
def reset_dialog(chat_id: int) -> None:
    DIALOGS[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
def hard_reset():
    DIALOGS = {}
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
async def backfill_dialog_history(chat_id: int, current_message_id: int | None = None) -> None:
    """
    При пустой памяти диалога подтягиваем последние N сообщений из Telegram,
    чтобы модель видела контекст до перезапуска.
    """
    if HISTORY_BACKFILL_MESSAGES <= 0:
        return
    if chat_id not in DIALOGS:
        reset_dialog(chat_id)
    history = DIALOGS.get(chat_id, [])
    if len(history) > 1:
        return  # уже есть накопленный диалог, бэкапы не нужны
    limit = HISTORY_BACKFILL_MESSAGES + (1 if current_message_id else 0)
    messages = await client.get_messages(chat_id, limit=limit)
    restored: List[dict] = []
    for m in reversed(messages):
        if current_message_id and getattr(m, "id", None) == current_message_id:
            continue
        text = (m.raw_text or "").strip()
        if not text:
            continue
        role = "assistant" if m.out else "user"
        restored.append({"role": role, "content": text})
    if restored:
        DIALOGS[chat_id].extend(restored)
        trim_history(chat_id)
def calc_confidence(scores: List[torch.Tensor], output_ids: torch.Tensor, prompt_len: int) -> float:
    """
    Оцениваем уверенность как среднее выбранных вероятностей токенов.
    """
    if not scores or output_ids is None or len(output_ids) <= prompt_len:
        return 0.0
    probs = []
    for step, logits in enumerate(scores):
        idx = prompt_len + step
        if idx >= len(output_ids):
            break
        token_id = int(output_ids[idx])
        with torch.no_grad():
            sm = torch.softmax(logits.float(), dim=-1)
        if sm.dim() == 2:
            sm = sm[0]
        if token_id < sm.shape[-1]:
            probs.append(sm[token_id])
    if not probs:
        return 0.0
    return float(torch.stack(probs).mean().item())
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
       # top_k=50,
        repetition_penalty=1.2,
       # no_repeat_ngram_size=4,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
def load_llm(lora_dir: str) -> None:
    global tokenizer, base_model, model
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
    tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    if tokenizer_local.pad_token_id is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token
    print(f"[LLM] Loading base model: {BASE_MODEL_ID} ({DEVICE}/{DTYPE})")
    base_local = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )
    base_local.resize_token_embeddings(len(tokenizer_local))
    print(f"[LLM] Loading LoRA adapter: {lora_dir}")
    model_local = PeftModel.from_pretrained(
        base_local,
        lora_dir,
        torch_dtype=DTYPE,
        device_map={"": DEVICE},
    )
    model_local.eval()
    tokenizer = tokenizer_local
    base_model = base_local
    model = model_local
    print("[LLM] Ready.")
@torch.inference_mode()
def llm_answer(chat_id: int, text: str, who: str) -> tuple[str, float]:
    if chat_id not in DIALOGS:
        reset_dialog(chat_id)
    DIALOGS[chat_id].append({"role": "user", "content": text})
    history = trim_history(chat_id)
    chat_ctx = build_chat_messages(history, who)
    if not chat_ctx:
        return "История пуста, отправьте сообщение ещё раз.", 0.0
    prompt_text = tokenizer.apply_chat_template(
        chat_ctx,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if len(prompt_ids) > MAX_CONTEXT_TOKENS:
        prompt_ids = prompt_ids[-MAX_CONTEXT_TOKENS:]
        # ???????????? ????? ??? ???????/???????????????, ?? ????????? ???? ?? prompt_ids
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    inputs = {
        "input_ids": torch.tensor([prompt_ids], device=model.device),
        "attention_mask": torch.ones(1, len(prompt_ids), device=model.device),
    }
    outputs = model.generate(
        **inputs,
        generation_config=current_gen_cfg(),
        return_dict_in_generate=True,
        output_scores=True,
    )
    output_ids = outputs.sequences[0]
    scores = list(outputs.scores or [])
    answer_ids = output_ids[len(prompt_ids):]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    confidence = calc_confidence(scores, output_ids, len(prompt_ids))
    if "😂" in answer:
        return llm_answer(chat_id, text, who)
    if "Вхвхв" in answer:
        return llm_answer(chat_id, text, who)
    if "Вхвхвхвх" in answer:
        return llm_answer(chat_id, text, who)
    if "Вхвх" in answer:
        return llm_answer(chat_id, text, who)
    if "ахах" in answer:
        return llm_answer(chat_id, text, who)
    DIALOGS[chat_id].append({"role": "assistant", "content": answer})
    trim_history(chat_id)
    return answer, confidence
# ───────────────────────────── TELETHON USERBOT ─────────────────────────────
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
target_chat_ids: set[int] = set()
async def keep_typing(chat_id, stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            async with client.action(chat_id, "typing"):
                # Telegram сам гасит статус через несколько секунд,
                # поэтому обновляем его периодически
                await asyncio.wait_for(stop_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        # просто продолжаем цикл, пока не будет stop_event.set()
        pass
    except asyncio.CancelledError:
        pass
async def load_filter_chat_ids(filter_id: int) -> set[int]:
    res = await client(GetDialogFiltersRequest())
    filters = res.filters if hasattr(res, "filters") else res
    f = next((x for x in filters if getattr(x, "id", None) == filter_id), None)
    if f is None:
        raise RuntimeError(f"Фильтр с id={filter_id} не найден")
    include_peers = getattr(f, "include_peers", None)
    if include_peers is None:
        raise RuntimeError(f"Фильтр id={filter_id} не содержит include_peers")
    ids = set()
    for p in include_peers:
        ent = await client.get_entity(p)
        ids.add(ent.id)
    return ids
async def is_admin(event: events.NewMessage.Event) -> bool:
    return bool(await get_whoo(event) == ADMIN_ID)
async def get_whoo(event: events.NewMessage.Event) -> str:
    """
    Надёжно извлекает имя отправителя:
    1. first_name + last_name
    2. username
    3. title (для чатов/каналов)
    4. fallback: 'Без имени'
    """
    # 1️⃣ если WHOO закреплён вручную
    if P.whoo_locked and (P.whoo_value or "").strip():
        return P.whoo_value.strip()
    try:
        # 2️⃣ ЯВНО резолвим entity по sender_id
        sender_id = event.sender_id
        if sender_id:
            entity = await client.get_entity(sender_id)
            # 👤 пользователь
            if hasattr(entity, "first_name"):
                parts = []
                if entity.first_name:
                    parts.append(entity.first_name)
                if getattr(entity, "last_name", None):
                    parts.append(entity.last_name)
                if parts:
                    return " ".join(parts)
                # username как fallback
                if getattr(entity, "username", None):
                    return entity.username
            # 👥 чат / канал
            if hasattr(entity, "title") and entity.title:
                return entity.title
    except Exception as e:
        print(f"[WHOO] resolve failed: {e}")
    # 3️⃣ последний fallback
    return P.whoo_default or "Без имени"
async def handle_command(event: events.NewMessage.Event, text: str) -> bool:
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
            "/stop\n",
            "/bred\n",
        )
        return True
    if cmd == "/params":
        await reply_safe(event, params_text())
        return True
    if cmd == "/clear" or cmd == "/стой":
        reset_dialog(event.chat_id)
        await reply_safe(event, "Контекст очищен для этого чата.")
        return True
    if not is_admin(event):
        await reply_safe(event, "Недостаточно прав.")
        return True
    if cmd == "/stop":
        global STOPED
        if STOPED:
            STOPED = False
        else:
            STOPED = True
        await reply_safe(event, f"Остановлен: {STOPED}")
        return True
    if cmd == "/bred":
        hard_reset()
        await reply_safe(event, "Диалог удален")
        return True
    if cmd == "/who":
        if args:
            P.whoo_value = " ".join(args).strip()
            P.whoo_locked = True
            await reply_safe(event, f"WHOO закреплён: {P.whoo_value}")
        else:
            P.whoo_locked = False
            await reply_safe(event, "WHOO не закреплён — имя будет браться из Telegram (или 'Без имени').")
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
    global target_chat_ids, COUNT
    if event.out:
        return
    chat_id = event.chat_id
    if chat_id not in target_chat_ids:
        return
    incoming_text = (event.raw_text or "").strip()
    if not incoming_text:
        return  # только текст
    # команды (без генерации)
    if await handle_command(event, incoming_text):
        return
    if STOPED:
        return
    if chat_id not in DIALOGS:
        reset_dialog(chat_id)
        await backfill_dialog_history(chat_id, current_message_id=getattr(event.message, "id", None))
    #print(event.message)
    who = await get_whoo(event)
    sender_id = getattr(event.sender, "id", 0) or 0
    await client.send_read_acknowledge(chat_id, getattr(event.sender, "id", 0))
    # лог входящего
    log_msg(chat_id, sender_id, who, "IN", incoming_text)
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(keep_typing(chat_id, stop_typing))
    try:
        async with client.action(chat_id, "typing"):
            answer, confidence = llm_answer(chat_id, incoming_text, who)
    finally:
        stop_typing.set()
        with contextlib.suppress(Exception):
            await typing_task
    # ??? ?????????? + ????????
    if confidence < MIN_CONFIDENCE:
        history = DIALOGS.get(chat_id, [])
        if history and history[-1].get("role") == "assistant" and history[-1].get("content") == answer:
            history.pop()
        log_msg(chat_id, sender_id, who, "OUT", f"[skip: low confidence {confidence:.3f}]")
        return
    log_msg(chat_id, sender_id, who, "OUT", f"[conf={confidence:.3f}] {answer}")
    await reply_safe(event, answer)
    COUNT = COUNT + 1
    if COUNT % 2 == 0:
        target_chat_ids = await load_filter_chat_ids(TARGET_FILTER_ID)
async def main():
    # 1) модель
    load_llm(P.lora_adapter_dir)
    # 2) telegram user auth
    await client.start()
    me = await client.get_me()
    print(f"[INFO] Telegram user: {me.id} {me.first_name} @{me.username}")
    # 3) target chats from filter
    global target_chat_ids
    target_chat_ids = await load_filter_chat_ids(TARGET_FILTER_ID)
    print(f"[INFO] Filter id={TARGET_FILTER_ID}: chats={len(target_chat_ids)}")
    print("[INFO] Listening...")
    await client.run_until_disconnected()
if __name__ == "__main__":
    asyncio.run(main())
