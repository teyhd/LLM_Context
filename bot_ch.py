import os
if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None and os.name != "nt":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import contextlib
import psutil
import random
import logging
import json
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
def _env_any(*names):
    for name in names:
        val = os.getenv(name)
        if val:
            return val
    return None

def _env_bool(*names, default: bool = False) -> bool:
    val = _env_any(*names)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

BASE_MODEL_ID     = _env_any("BASE_MODEL_ID", "base_model_id") or "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_DIR = _env_any("MODEL_DIR", "model_dir") or "models/vlad_20260208_104709"
PRIMARY_ADAPTER = _env_any("PRIMARY_ADAPTER", "primary_adapter") or "final_adapter"
LORA_ADAPTER_DIR  = _env_any("LORA_ADAPTER_DIR", "lora_adapter_dir")
TRAIN_CONFIG_PATH = _env_any("TRAIN_CONFIG_PATH", "train_config_path")
if not LORA_ADAPTER_DIR:
    LORA_ADAPTER_DIR = str(Path(MODEL_DIR) / PRIMARY_ADAPTER)

SYSTEM_PROMPT = _env_any("SYSTEM_PROMPT", "system_prompt")
USER_INSTRUCTION_TEMPLATE = _env_any("USER_INSTRUCTION_TEMPLATE", "user_instruction_template")
MAX_CONTEXT_TOKENS = _env_any("MAX_CONTEXT_TOKENS", "max_context_tokens")
MAX_HISTORY_MESSAGES = int(_env_any("MAX_HISTORY_MESSAGES", "max_history_messages") or "40")
ADMIN_ID          = int(_env_any("ADMIN_ID", "admin_id") or "304622290")
ADMIN_CHAT_ID     = int(_env_any("ADMIN_CHAT_ID", "admin_chat_id") or "304622290")
TELEGRAM_TOKEN    = _env_any("TELEGRAM_TOKEN", "telegram_token") or "667589363:AAFIFSIh3Yyy2dyratXGwaCP2bAkc8DI-tY"

PROMPT_CONFIG_PATH = _env_any("PROMPT_CONFIG_PATH", "prompt_config_path") or "config/prompt_config.json"
if os.path.exists(PROMPT_CONFIG_PATH):
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            _pdata = json.load(f)
        SYSTEM_PROMPT = _pdata.get("system_prompt") or SYSTEM_PROMPT
        USER_INSTRUCTION_TEMPLATE = _pdata.get("user_instruction_template") or USER_INSTRUCTION_TEMPLATE
    except Exception:
        pass

def _instruction_prefix(template: str) -> str:
    if "{who}" in template:
        return template.split("{who}", 1)[0].strip()
    return "\u0418\u043c\u044f \u0441\u043e\u0431\u0435\u0441\u0435\u0434\u043d\u0438\u043a\u0430:"

INSTR_PREFIX = _instruction_prefix(USER_INSTRUCTION_TEMPLATE)

DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE             = torch.float16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS    = _env_any("MAX_NEW_TOKENS", "max_new_tokens")
TEMPERATURE       = _env_any("TEMPERATURE", "temperature")
TOP_P             = _env_any("TOP_P", "top_p")
REPETITION_PENALTY= _env_any("REPETITION_PENALTY", "repetition_penalty")
NO_REPEAT_NGRAM_SIZE= _env_any("NO_REPEAT_NGRAM_SIZE", "no_repeat_ngram_size")
DO_SAMPLE        = _env_bool("DO_SAMPLE", "do_sample", default=False)

WHOO = _env_any("DEFAULT_WHO", "who") or "\u0410\u043b\u0438\u0441\u0430 \u042e\u0440\u044c\u0435\u0432\u043d\u0430"

LOG_FILE = "bot_ch.log"
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

def list_adapter_dirs(model_dir: str, fallback_adapter: str | None = None) -> list[tuple[str, Path]]:
    base = Path(model_dir)
    adapters: list[tuple[str, Path]] = []
    final_dir = base / "final_adapter"
    if final_dir.exists():
        adapters.append(("final_adapter", final_dir))
    def _ckpt_key(path: Path):
        name = path.name
        try:
            return int(name.split("-")[-1])
        except Exception:
            return name

    for ckpt in sorted(base.glob("checkpoint-*"), key=_ckpt_key):
        adapter_dir = ckpt / "adapter"
        if adapter_dir.exists():
            adapters.append((ckpt.name, adapter_dir))
    if not adapters and fallback_adapter:
        p = Path(fallback_adapter)
        if p.exists():
            name = p.parent.name if p.name == "adapter" else p.name
            adapters.append((name, p))
    return adapters


# ─────────────────── Загрузка модели ────────────────────────
# Model loading

def _resolve_train_config_path() -> Path | None:
    if TRAIN_CONFIG_PATH:
        p = Path(TRAIN_CONFIG_PATH)
        return p if p.exists() else None
    try:
        model_dir = Path(MODEL_DIR)
        candidate = model_dir / "train_config.json"
        if candidate.exists():
            return candidate
        lora_dir = Path(LORA_ADAPTER_DIR)
        candidate = lora_dir.parent / "train_config.json"
        return candidate if candidate.exists() else None
    except Exception:
        return None

def _apply_train_defaults() -> None:
    global MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE
    cfg_path = _resolve_train_config_path()
    if not cfg_path:
        return
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return
    if MAX_CONTEXT_TOKENS is None:
        MAX_CONTEXT_TOKENS = str(cfg.get("max_seq_len") or cfg.get("max_context_tokens") or "2048")
    if MAX_NEW_TOKENS is None:
        MAX_NEW_TOKENS = str(cfg.get("max_gen_tokens") or cfg.get("max_answer_tokens") or "128")
    if TEMPERATURE is None:
        TEMPERATURE = str(cfg.get("temperature") or "0.6")
    if TOP_P is None:
        TOP_P = str(cfg.get("top_p") or "0.9")
    if REPETITION_PENALTY is None:
        REPETITION_PENALTY = str(cfg.get("repetition_penalty") or "1.1")
    if NO_REPEAT_NGRAM_SIZE is None:
        NO_REPEAT_NGRAM_SIZE = str(cfg.get("no_repeat_ngram_size") or "3")

_apply_train_defaults()

MAX_CONTEXT_TOKENS = int(MAX_CONTEXT_TOKENS or "2048")
MAX_NEW_TOKENS = int(MAX_NEW_TOKENS or "128")
TEMPERATURE = float(TEMPERATURE or "0.6")
TOP_P = float(TOP_P or "0.9")
REPETITION_PENALTY = float(REPETITION_PENALTY or "1.1")
NO_REPEAT_NGRAM_SIZE = int(NO_REPEAT_NGRAM_SIZE or "3")

tokenizer = None
base_model = None
model = None
ADAPTER_ORDER: list[str] = []
PRIMARY_ADAPTER_DISPLAY = None
DISPLAY_TO_MODEL: dict[str, str] = {}

def _tokenizer_source() -> str:
    final_dir = Path(MODEL_DIR) / "final_adapter"
    for base in (final_dir, Path(LORA_ADAPTER_DIR)):
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json"):
            if (base / name).exists():
                return str(base)
    return BASE_MODEL_ID

def load_llm() -> None:
    global tokenizer, base_model, model, ADAPTER_ORDER, PRIMARY_ADAPTER_DISPLAY, DISPLAY_TO_MODEL
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

    adapter_entries = list_adapter_dirs(MODEL_DIR, LORA_ADAPTER_DIR)
    if not adapter_entries:
        raise RuntimeError(f"No adapters found in {MODEL_DIR!r} and LORA_ADAPTER_DIR={LORA_ADAPTER_DIR!r}")

    print("Loading tokenizer...")
    tokenizer_src = _tokenizer_source()
    tokenizer_local = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
    if tokenizer_local.pad_token_id is None:
        tokenizer_local.pad_token = tokenizer_local.eos_token

    print(f"Loading base model: {BASE_MODEL_ID} ({DEVICE}/{DTYPE})")
    base_local = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=DTYPE,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    base_local.resize_token_embeddings(len(tokenizer_local))

    first_name, first_path = adapter_entries[0]
    print(f"Loading LoRA adapter: {first_path}")
    model_local = PeftModel.from_pretrained(
        base_local,
        str(first_path),
        dtype=DTYPE,
        device_map={"": DEVICE},
    )
    display_to_model = {first_name: "default"}
    for name, path in adapter_entries[1:]:
        model_local.load_adapter(str(path), adapter_name=name)
        display_to_model[name] = name

    primary = PRIMARY_ADAPTER or adapter_entries[0][0]
    if primary not in display_to_model:
        primary = adapter_entries[0][0]
    model_local.set_adapter(display_to_model[primary])
    model_local.eval()

    tokenizer = tokenizer_local
    base_model = base_local
    model = model_local
    ADAPTER_ORDER = [name for name, _ in adapter_entries]
    PRIMARY_ADAPTER_DISPLAY = primary
    DISPLAY_TO_MODEL = display_to_model
    print("Ready.")

# Model initialization
load_llm()

_gen_kwargs = dict(
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REPETITION_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    do_sample=DO_SAMPLE,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)
if DO_SAMPLE:
    _gen_kwargs["temperature"] = TEMPERATURE
    _gen_kwargs["top_p"] = TOP_P
GEN_CFG = GenerationConfig(**_gen_kwargs)

# ──────────────────── Telegram-бот ──────────────────────────
bot = TeleBot(TELEGRAM_TOKEN, parse_mode=None)

DIALOGS = {}                 # user_id → list[dict(role, content)]

def reset_dialog(uid: int):
    DIALOGS[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]

def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))

def trim_history(uid: int) -> list[dict]:
    history = DIALOGS.get(uid, [])
    if not history:
        return []
    head = history[:1] if history[0].get("role") == "system" else []
    tail = history[1:]
    if len(tail) > MAX_HISTORY_MESSAGES:
        tail = tail[-MAX_HISTORY_MESSAGES:]
    if MAX_CONTEXT_TOKENS and tokenizer:
        tokens = [_count_tokens(m.get("content", "")) for m in tail]
        total = sum(tokens)
        start_idx = 0
        while total > MAX_CONTEXT_TOKENS and start_idx < len(tail):
            total -= tokens[start_idx]
            start_idx += 1
        tail = tail[start_idx:]
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
        if role == "user" and INSTR_PREFIX and INSTR_PREFIX not in content:
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
def _generate_from_prompt(prompt_ids: list[int]) -> str:
    inputs = {
        "input_ids": torch.tensor([prompt_ids], device=model.device),
        "attention_mask": torch.ones(1, len(prompt_ids), device=model.device),
    }
    output_ids = model.generate(**inputs, generation_config=GEN_CFG)[0]
    answer_ids = output_ids[len(prompt_ids):]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

@torch.inference_mode()
def llm_answer(user_id: int, text: str, who: str) -> str:
    if user_id not in DIALOGS:
        reset_dialog(user_id)
    DIALOGS[user_id].append({"role": "user", "content": text})
    history = trim_history(user_id)
    chat_ctx = build_chat_messages(history, who)
    if not chat_ctx:
        return "??????? ?????, ????????? ????????? ??? ???."

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

    log_context(user_id, who, history, prompt_text, len(prompt_ids))

    answers: list[tuple[str, str]] = []
    primary_answer: str | None = None
    for display_name in ADAPTER_ORDER:
        adapter_name = DISPLAY_TO_MODEL.get(display_name)
        if not adapter_name:
            continue
        model.set_adapter(adapter_name)
        answer = _generate_from_prompt(prompt_ids)
        answers.append((display_name, answer))
        if display_name == PRIMARY_ADAPTER_DISPLAY:
            primary_answer = answer
        logging.info("ANSWER adapter=%s uid=%s len=%d text=%s", display_name, user_id, len(answer), answer)

    if primary_answer is None and answers:
        primary_answer = answers[0][1]
    if primary_answer:
        DIALOGS[user_id].append({"role": "assistant", "content": primary_answer})
        trim_history(user_id)

    if not answers:
        return "Ответов не получено."
    return "\n\n".join([f"{name}\nОтвет: {ans}" for name, ans in answers])

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
