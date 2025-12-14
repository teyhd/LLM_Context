# ───────────────────────────────────────────────────────────────────────
import os, csv, json, math, random, logging, warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.serialization
import numpy.core.multiarray

torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct])

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, TaskType

# ─── ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ────────────────────────────────────────────

MODEL_ID               = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_DIR               = "data/output"
TRAIN_FILE             = "train.jsonl"
VAL_FILE               = "valid.jsonl"

OUTPUT_DIR             = "models"
# фиксированное имя прогона, совпадает с ботом
RUN_NAME               = "vlad4"

NOTIFY_URL             = "http://home.teyhd.ru:3334/"

NUM_EPOCHS             = 3
BATCH_SIZE             = 2
GRAD_ACC               = 4
LEARNING_RATE          = 2e-5

WARMUP_FRAC            = 0.05
WEIGHT_DECAY           = 0.01
MAX_SEQ_LEN            = 2048
MAX_GRAD_NORM          = 0.3

LORA_R                 = 32#8
LORA_ALPHA             = 64#16
LORA_DROPOUT           = 0.2

# Блоки, к которым будет применяться LoRA (типичный набор для Mistral)
TARGET_MODULES = [
    "q_proj",
   # "k_proj",
    "v_proj",
    #"o_proj",
    "gate_proj",
   # "up_proj",
   # "down_proj",
]

SAVE_STEPS             = 200
EVAL_STEPS             = 200
SAVE_LIMIT             = 4
EARLY_PATIENCE         = 20   # реальная ранняя остановка

LOG_STEPS              = 5
CSV_METRICS            = True
CSV_FILE               = "metrics.csv"
LOSS_ALERT             = 5.0

GEN_INTERVAL           = 25
MAX_GEN_TOKENS         = 128
TEMPERATURE            = 0.4
TOP_P                  = 0.7

ANALYTICS_STEPS        = 5  # шаги, через которые шлём короткую сводку

SEED                   = 28

USE_FP16               = True
USE_BF16               = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
GRADIENT_CHECKPOINTING = True
REPORT_TO_WANDB        = False

# reproducibility & env
random.seed(SEED)
torch.manual_seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

# ─── Уведомления ──────────────────────────────────────────────────────
def notify(msg: str) -> None:
    if not NOTIFY_URL:
        return
    try:
        import requests
        print(msg)
        requests.get(NOTIFY_URL, params={"msg": f"SRV: {msg[:1000]}"})
    except Exception as e:                                     # noqa: BLE001
        logging.warning(f"notify failed: {e}")

# ─── Логирование ──────────────────────────────────────────────────────
OUT_DIR = Path(OUTPUT_DIR) / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "train.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, "w", "utf-8"), logging.StreamHandler()],
)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Tokenizer ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id

# ─── Helpers ──────────────────────────────────────────────────────────
def read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def format_chat(messages: List[Dict]) -> (str, str):
    """
    Готовим prompt через штатный шаблон Mistral, ответ — последний assistant.
    """
    ass_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not ass_indices:
        return "", ""

    last_ass_idx = ass_indices[-1]
    answer = messages[last_ass_idx]["content"].strip()
    if not answer:
        return "", ""

    def to_alternating(msgs: List[Dict]) -> List[Dict]:
        # Склеиваем подряд идущие роли и убираем пустое содержимое
        merged = []
        for m in msgs:
            content = m["content"].strip()
            if not content:
                continue
            role = m["role"]
            if merged and merged[-1]["role"] == role:
                merged[-1]["content"] += "\n" + content
            else:
                merged.append({"role": role, "content": content})

        # Оставляем только первый system, если он есть
        has_system = merged and merged[0]["role"] == "system"
        core = merged[1:] if has_system else merged

        # Убираем ведущие assistant, чтобы диалог начинался с user
        while core and core[0]["role"] == "assistant":
            core = core[1:]

        # Гарантируем чередование user/assistant
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

        # Обрезаем хвост, чтобы закончить на user (требование шаблона)
        while alternated and alternated[-1]["role"] != "user":
            alternated.pop()
        return alternated

    chat_ctx = to_alternating(messages[:last_ass_idx])
    if not chat_ctx:
        return "", ""

    prompt = tokenizer.apply_chat_template(
        chat_ctx,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, answer

def build_dataset(raw: List[Dict]) -> Dataset:
    samples = []
    for rec in raw:
        msgs = rec["messages"]
        # Требуем, чтобы последний в списке был ассистентом
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            continue

        prompt, answer = format_chat(msgs)
        if not prompt or not answer:
            continue

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        ).input_ids
        answer_ids = tokenizer(
            answer,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        ).input_ids

        max_prompt_len = MAX_SEQ_LEN - len(answer_ids) - 1
        if max_prompt_len <= 0:
            # Ответ сам по себе больше лимита — режем его, чтобы не терять выборку
            answer_ids = answer_ids[: MAX_SEQ_LEN - 1]
            prompt_chunks = [[]]
        else:
            # Разбиваем длинный prompt на окна, чтобы не терять контекст
            prompt_chunks = []
            if not prompt_ids:
                prompt_chunks = [[]]
            elif len(prompt_ids) <= max_prompt_len:
                prompt_chunks = [prompt_ids]
            else:
                stride = max(max_prompt_len // 2, 1)
                n = len(prompt_ids)
                start = 0
                while start < n:
                    chunk = prompt_ids[start:start + max_prompt_len]
                    prompt_chunks.append(chunk)
                    if start + max_prompt_len >= n:
                        break
                    start += stride
                tail = prompt_ids[-max_prompt_len:]
                if prompt_chunks and prompt_chunks[-1] != tail:
                    prompt_chunks.append(tail)

        for chunk in prompt_chunks:
            input_ids = chunk + answer_ids + [eos_id]
            if len(input_ids) > MAX_SEQ_LEN:
                continue
            labels = [-100] * len(chunk) + answer_ids + [eos_id]
            samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    if not samples:
        logging.warning("⚠️ build_dataset: no valid samples constructed")
        return Dataset.from_list([])

    return Dataset.from_list(samples).with_format("torch")

# ─── Data ─────────────────────────────────────────────────────────────
train_raw = read_jsonl(Path(DATA_DIR) / TRAIN_FILE)
val_raw   = read_jsonl(Path(DATA_DIR) / VAL_FILE)

train_ds  = build_dataset(train_raw)
val_ds    = build_dataset(val_raw)

logging.info(f"📊 samples → {len(train_ds)} train • {len(val_ds)} val")

# ─── Model + LoRA ─────────────────────────────────────────────────────
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if USE_BF16 else (torch.float16 if USE_FP16 else torch.float32),
    device_map="auto",
)
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.use_cache = False
if GRADIENT_CHECKPOINTING:
    base_model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

# ─── Callbacks ────────────────────────────────────────────────────────
class CSVLogger(TrainerCallback):
    def __init__(self):
        self.columns = ["step"]
        self.header_written = False

    def _read_existing_header(self, csv_path: Path) -> None:
        if self.header_written or not csv_path.exists():
            return
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)
            if header:
                self.columns = list(header)
                self.header_written = True
        except Exception as e:  # noqa: BLE001
            logging.warning("csv header read failed: %s", e)

    def on_log(self, args, state, control, logs=None, **kw):
        if not CSV_METRICS or not logs:
            return
        csv_path = OUT_DIR / CSV_FILE
        self._read_existing_header(csv_path)
        for key in logs:
            if key not in self.columns:
                self.columns.append(key)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.columns)
            if not self.header_written:
                w.writeheader()
                self.header_written = True
            row = {"step": state.global_step}
            for key in self.columns:
                if key == "step":
                    continue
                row[key] = logs.get(key)
            w.writerow(row)
        if logs.get("loss", 0) > LOSS_ALERT:
            notify(f"⚠️ loss {state.global_step}: {logs['loss']:.2f}")
        if state.global_step and state.global_step % ANALYTICS_STEPS == 0:
            loss = logs.get("loss")
            eval_loss = logs.get("eval_loss")
            lr = logs.get("learning_rate")
            parts = [f"step {state.global_step}"]
            if loss is not None:
                parts.append(f"loss={loss:.3f}")
            if eval_loss is not None:
                parts.append(f"eval={eval_loss:.3f}")
            if lr is not None:
                parts.append(f"lr={lr:.2e}")
            msg = " | ".join(parts)
            logging.info(msg)
            notify(msg)

class SaveAdapter(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        p = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "adapter"
        p.mkdir(parents=True, exist_ok=True)
        kw["model"].save_pretrained(p)
        logging.info(f"💾 adapter saved at step {state.global_step}")
        notify(f"💾 checkpoint {state.global_step}")

class StartEndNotify(TrainerCallback):
    def on_train_begin(self, *a, **kw):
        notify("🚀 Старт обучения")

    def on_train_end(self, *a, **kw):
        notify("✅ Обучение завершено")

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics and "eval_loss" in metrics:
            notify(f"📉 eval {state.global_step}: {metrics['eval_loss']:.4f}")

class RandomGenerateNotify(TrainerCallback):
    def __init__(self, data: List[Dict], interval: int = GEN_INTERVAL):
        self.data = data
        self.interval = interval

    def safe_generate(self, prompt_text: str) -> str:
        try:
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            )
            prompt_ids = enc.input_ids
            attn_mask = enc.attention_mask
            if prompt_ids.numel() == 0:
                return "[SKIP: empty prompt]"

            prompt_ids = prompt_ids.to(model.device)
            attn_mask = attn_mask.to(model.device)
            model.eval()

            with torch.no_grad():
                out = model.generate(
                    prompt_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=MAX_GEN_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=eos_id,
                    eos_token_id=eos_id,
                    do_sample=True,
                )

            return tokenizer.decode(
                out[0][prompt_ids.shape[-1]:],
                skip_special_tokens=True,
            )

        except RuntimeError as e:
            if "CUDA error" in str(e) or "device-side assert" in str(e):
                logging.warning(f"[GEN FAIL] CUDA error: {e}")
                notify(f"⚠️ Ошибка генерации: {e}")
                return "[GENERATION FAILED]"
            raise

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return

        rec = random.choice(self.data)
        prompt, gold = format_chat(rec["messages"])
        if not prompt:
            return

        # Берём последний user перед ответом для отчёта
        last_user = ""
        for m in reversed(rec["messages"]):
            if m.get("role") == "user":
                last_user = m.get("content", "").strip()
                break

        gen = self.safe_generate(prompt)
        notify(
            f"🎙 {state.global_step}\nUSER: {last_user}\nGOLD: {gold}\nGEN: {gen}\nPROMPT: {prompt}"
        )

# ─── TrainingArguments ───────────────────────────────────────────────
updates = math.ceil(len(train_ds) / BATCH_SIZE / GRAD_ACC) * NUM_EPOCHS if len(train_ds) > 0 else 0
warmup  = max(10, int(updates * WARMUP_FRAC)) if updates > 0 else 0

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    run_name=RUN_NAME,
    seed=SEED,

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=warmup,
    lr_scheduler_type="cosine",

    fp16=USE_FP16 and not USE_BF16,
    bf16=USE_BF16,
    max_grad_norm=MAX_GRAD_NORM,

    logging_steps=LOG_STEPS,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_LIMIT,
    load_best_model_at_end=True,
    group_by_length=True,
    eval_accumulation_steps=4,

    optim="adamw_torch_fused",
    report_to=[] if not REPORT_TO_WANDB else ["wandb"],
    save_safetensors=True,
)

# ─── Data Collator ───────────────────────────────────────────────────
def collate(batch):
    pad_id = tokenizer.pad_token_id
    input_ids      = torch.nn.utils.rnn.pad_sequence(
        [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
    )
    labels         = torch.nn.utils.rnn.pad_sequence(
        [x["labels"] for x in batch], batch_first=True, padding_value=-100
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ─── Trainer ─────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
    callbacks=[
        CSVLogger(),
        SaveAdapter(),
        StartEndNotify(),
        RandomGenerateNotify(train_raw),
        EarlyStoppingCallback(early_stopping_patience=EARLY_PATIENCE),
    ],
)

# ─── Run ─────────────────────────────────────────────────────────────
ckpts = sorted(
    OUT_DIR.glob("checkpoint-*"),
    key=lambda p: int(p.name.split("-")[-1]),
)
resume_ckpt = str(ckpts[-1]) if ckpts else None
if resume_ckpt and not Path(resume_ckpt).exists():
    resume_ckpt = None

notify(f"Resume {resume_ckpt}" if resume_ckpt else "New run")
trainer.train(resume_from_checkpoint=resume_ckpt)

# ─── Save final adapter ──────────────────────────────────────────────
adapter_dir = OUT_DIR / "final_adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
logging.info(f"Adapter saved to {adapter_dir}")
notify("🎉 Финальный адаптер сохранён")
