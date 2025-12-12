import os
import gc
import psutil
import random
import logging
import datetime as dt
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from telebot import TeleBot, apihelper
from pynvml import (nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
                    nvmlDeviceGetTemperature, nvmlDeviceGetMemoryInfo, nvmlShutdown)

# ──────────────────────── Константы ─────────────────────────
BASE_MODEL_ID     = "mistralai/Mistral-7B-Instruct-v0.3"
LORA_ADAPTER_DIR  = "models/vlad3/checkpoint-1000"

SYSTEM_PROMPT    = "Ты Влад. Ты дружелюбный и лаконичный.\nГлавный фокус — переписка: отвечай по делу, без лишней воды."
USER_INSTRUCTION_TEMPLATE = "Имя собеседника: {who}. Напиши ответ на сообщение: {text}"
MAX_CONTEXT_TOKENS = 2048
MAX_HISTORY_MESSAGES = 40
ADMIN_ID          = 304622290
ADMIN_CHAT_ID     = 304622290  
TELEGRAM_TOKEN    = "667589363:AAFIFSIh3Yyy2dyratXGwaCP2bAkc8DI-tY"

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS    = 128  
TEMPERATURE       = 0.4
TOP_P             = 0.7        
WHOO = "Алиса Юрьевна"

LOG_FILE = "bot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, "a", "utf-8"),
        logging.StreamHandler(),
    ],
)

# ─────────────────────── Помощники ──────────────────────────
def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**20

def gpu_info() -> str:
    if DEVICE != "cuda":
        return "GPU отсутствует"
    try:
        nvmlInit()
        h       = nvmlDeviceGetHandleByIndex(0)
        util    = nvmlDeviceGetUtilizationRates(h)
        temp    = nvmlDeviceGetTemperature(h, 0)
        mem     = nvmlDeviceGetMemoryInfo(h)
        nvmlShutdown()
        mb = lambda x: x / 2**20
        return (f"GPU util: {util.gpu}%\n"
                f"t°: {temp}°C\n"
                f"mem used: {mb(mem.used):.0f}/{mb(mem.total):.0f} MB")
    except Exception as e:
        return f"NVML-ошибка: {e}"

def safe_send(bot: TeleBot, chat_id: int, text: str, *args, **kw):
    """Делит длинный текст на части ≤ 4096 симв. (без Markdown)."""
    for chunk in (text[i:i+4000] for i in range(0, len(text), 4000)):
        bot.send_message(chat_id, chunk, *args, **kw)

# ─────────────────── Загрузка модели ────────────────────────
print("Загружаю базовую модель…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
base_model.resize_token_embeddings(len(tokenizer))          # safety

print("Подключаю LoRA-адаптер…")
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER_DIR,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
)
#model = base_model
model.eval()

GEN_CFG = GenerationConfig(
    max_new_tokens = MAX_NEW_TOKENS,
    temperature    = TEMPERATURE,
    top_p          = TOP_P,
    do_sample      = True,
    eos_token_id   = tokenizer.eos_token_id,
    pad_token_id   = tokenizer.pad_token_id,
)

# ──────────────────── Telegram-бот ──────────────────────────
bot = TeleBot(TELEGRAM_TOKEN, parse_mode=None)

DIALOGS = {}                 # user_id → list[dict(role, content)]

def reset_dialog(uid: int):
    DIALOGS[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]

def trim_history(uid: int) -> list[dict]:
    history = DIALOGS.get(uid, [])
    if not history:
        return []
    head = history[:1] if history[0].get("role") == "system" else []
    tail = history[1:]
    if len(tail) > MAX_HISTORY_MESSAGES:
        tail = tail[-MAX_HISTORY_MESSAGES:]
    history = head + tail
    DIALOGS[uid] = history
    return history

def build_chat_messages(messages: list[dict], who: str) -> list[dict]:
    """Готовим историю в формате chat_template Mistral и приводим к чередованию user/assistant."""
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

def log_context(uid: int, who: str, history: list[dict], prompt: str, prompt_tokens: int) -> None:
    """Логируем весь контекст перед генерацией, чтобы видеть, что ушло в модель."""
    try:
        logging.info(
            "CTX uid=%s who=%s msgs=%d prompt_tokens=%d\n%s",
            uid, who, len(history), prompt_tokens, prompt,
        )
    except Exception as e:  # noqa: BLE001
        logging.warning("log_context failed: %s", e)

@torch.inference_mode()
def llm_answer(user_id: int, text: str, who: str) -> str:
    if user_id not in DIALOGS:
        reset_dialog(user_id)
    DIALOGS[user_id].append({"role": "user", "content": text})
    history = trim_history(user_id)
    chat_ctx = build_chat_messages(history, who)
    if not chat_ctx:
        return "История пуста, отправьте сообщение ещё раз."

    # Строим prompt как в обучении и вручную обрезаем историю справа, чтобы не терять свежий контекст
    prompt_text = tokenizer.apply_chat_template(
        chat_ctx,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    ).input_ids
    if len(prompt_ids) > MAX_CONTEXT_TOKENS:
        prompt_ids = prompt_ids[-MAX_CONTEXT_TOKENS:]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

    inputs = {
        "input_ids": torch.tensor([prompt_ids], device=model.device),
        "attention_mask": torch.ones(1, len(prompt_ids), device=model.device),
    }
    log_context(user_id, who, history, prompt_text, len(prompt_ids))

    output_ids = model.generate(**inputs, generation_config=GEN_CFG)[0]
    answer_ids = output_ids[len(prompt_ids):]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    DIALOGS[user_id].append({"role": "assistant", "content": answer})
    trim_history(user_id)
    logging.info("ANSWER uid=%s len=%d text=%s", user_id, len(answer), answer)
    return answer

# ──────────────── Обработчики команд ────────────────────────
@bot.message_handler(commands=["start", "help"])
def cmd_help(msg):
    safe_send(bot, msg.chat.id,
              "/help – команды\n/clear – очистить контекст\n/info – ресурсы\n/kill – выключить (адм.)")

@bot.message_handler(commands=["clear"])
def cmd_clear(msg):
    reset_dialog(msg.from_user.id)
    bot.reply_to(msg, "Контекст очищен.")

@bot.message_handler(commands=["info"])
def cmd_info(msg):
    mem = ram_mb()
    gpu = gpu_info()
    safe_send(bot, msg.chat.id, f"RAM: {mem:.0f} MB\n{gpu}")


@bot.message_handler(commands=["who"])
def cmd_who(message):
    global WHOO
    response = f"Такой команды нет!\nДопустимые команды:\n/help - Допустимые команды\n/login pass - Авторизоваться\n/clear - Отчистка истории диалога"
    cmd = message.text.split()[0]
    args = message.text.split()[1:] 
    if cmd == '/who':
        response = f"Вы отправили команду /who с аргументами: {' '.join(args)}"
        if len(args)>0:
                WHOO = ' '.join(args)
                response = f"Вы {WHOO}"
        else:
                response = f"Вы отправили команду /who с аргументами: {' '.join(args)}"
    safe_send(bot, message.chat.id, response)

@bot.message_handler(commands=["kill"])
def cmd_kill(msg):
    if msg.from_user.id != ADMIN_ID:
        bot.reply_to(msg, "Недостаточно прав.")
        return
    bot.reply_to(msg, "Выключаюсь…")
    bot.stop_polling()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# ────────────────── Основной текстовый хэндлер ──────────────
@bot.message_handler(content_types=["text"])
def handle_text(msg):
   # print(msg)
    uid = msg.from_user.id
    who = WHOO or msg.from_user.first_name or "user"
    logging.info("IN uid=%s who=%s text=%s", uid, who, msg.text)
    #if uid != ADMIN_ID:
       # bot.reply_to(msg, "Бот доступен только владельцу.")
       # return

    try:
        start = dt.datetime.now()
        answer = llm_answer(uid, msg.text, who)
        safe_send(bot, uid, answer)
        dur = dt.datetime.now() - start
        logging.info("OUT uid=%s dur=%.1fs len=%d", uid, dur.total_seconds(), len(answer))
    except Exception as e:
        bot.reply_to(msg, f"⚠️ Ошибка: {e}")
        raise

# ───────────────────────── Run ──────────────────────────────
if __name__ == "__main__":
    bot.send_message(ADMIN_ID, "🤖 Запустился и готов к работе.")
    bot.infinity_polling(skip_pending=True)
