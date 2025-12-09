import os
import gc
import psutil
import random
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

LORA_ADAPTER_DIR  = "models/vlad/final_adapter"

SYSTEM_PROMPT    = "Тебя зовут Влад. Ты человек, эксперт в сфере IT. Пиши короткие ответы, мотивируй продолжать диалог."
USER_INSTRUCTION_TEMPLATE = "Имя пользователя: {who}. Напиши ответ на сообщение: {text}"
MAX_CONTEXT_TOKENS = 2048
MAX_HISTORY_MESSAGES = 40
ADMIN_ID          = 304622290
ADMIN_CHAT_ID     = 304622290  
TELEGRAM_TOKEN    = "667589363:AAFIFSIh3Yyy2dyratXGwaCP2bAkc8DI-tY"

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS    = 128  
TEMPERATURE            = 0.7
TOP_P                  = 0.8        
WHOO = "Наталья Соина"

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
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
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

def build_prompt(current_user_text: str, who: str) -> str:
    """Форматирует запрос так же, как в тренировочном датасете."""
    user_text = USER_INSTRUCTION_TEMPLATE.format(who=who, text=current_user_text)
    parts = [
        f"[system]{SYSTEM_PROMPT}[/system]",
        f"[user]{user_text}[/user]",
    ]
    return "\n".join(parts) + "\n[assistant]"

@torch.inference_mode()
def llm_answer(user_id: int, text: str, who: str) -> str:
    if user_id not in DIALOGS:
        reset_dialog(user_id)
    DIALOGS[user_id].append({"role": "user", "content": text})
    trim_history(user_id)
    prompt = build_prompt(text, who)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_TOKENS,
    ).to(model.device)
    output_ids = model.generate(**inputs, generation_config=GEN_CFG)[0]
    answer_ids = output_ids[inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    DIALOGS[user_id].append({"role": "assistant", "content": answer})
    trim_history(user_id)
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
    print(f"[{uid}:{who}]\n{msg.text}")
    #if uid != ADMIN_ID:
       # bot.reply_to(msg, "Бот доступен только владельцу.")
       # return

    try:
        start = dt.datetime.now()
        answer = llm_answer(uid, msg.text,who)
        safe_send(bot, uid, answer)
        dur = dt.datetime.now() - start
        print(f"[{uid}] {dur.total_seconds():.1f}s ⇒ {len(answer)} симв.")
    except Exception as e:
        bot.reply_to(msg, f"⚠️ Ошибка: {e}")
        raise

# ───────────────────────── Run ──────────────────────────────
if __name__ == "__main__":
    bot.send_message(ADMIN_ID, "🤖 Запустился и готов к работе.")
    bot.infinity_polling(skip_pending=True)
