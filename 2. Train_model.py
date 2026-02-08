import argparse
import json
import logging
import math
import os
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import time
import inspect
import platform
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Ensure torch.utils.checkpoint always passes use_reentrant explicitly
try:
    from torch.utils import checkpoint as _ckpt
    if "use_reentrant" in inspect.signature(_ckpt.checkpoint).parameters:
        _orig_checkpoint = _ckpt.checkpoint
        def _checkpoint(*args, **kwargs):
            kwargs.setdefault("use_reentrant", False)
            return _orig_checkpoint(*args, **kwargs)
        _ckpt.checkpoint = _checkpoint
except Exception:
    pass

# ========================
# Default config
# ========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "profile": "quality",
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    "data_dir": "data/output",
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    "output_dir": "models",
    "run_name": "vlad",
    "run_dir": "",
    "resume": True,
    "resume_optimizer": True,
    "seed": 42,
    "num_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_len": 2048,
    "max_answer_tokens": 1024,
    "max_grad_norm": 0.3,
    "use_fp16": True,
    "use_bf16": "auto",
    "gradient_checkpointing": True,
    "use_4bit": True,
    "auto_qlora": True,
    "vram_gb_threshold": 28,
    "use_flash_attn": "auto",
    "pack_samples": True,
    "auto_oom_recovery": True,
    "oom_retries": 1,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "save_steps": 25,
    "eval_steps": 25,
    "save_total_limit": 4,
    "logging_steps": 25,
    "early_stopping_patience": 20,
    "report_to": [],
    "optim": "auto",
    "group_by_length": True,
    "generate_interval": 0,
    "max_gen_tokens": 128,
    "temperature": 0.6,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "sample_prompts": [],
    "assistant_turns": "last",  # all | last
    "max_samples_per_dialog": 0,
    "notify_url": "",
    "notify_interval": 25,
    "notify_sample_interval": 25,
    "notify_sample_do_sample": False,
    "notify_dialog_interval": 25,
    "notify_dialogs": [],
    "notify_dialog_turns": 4,
    "alert_loss_spike_ratio": 1.5,
    "alert_grad_norm": 10.0,
    "save_merged": False,
    "sanity_dialogs": [],
}

PROFILES: Dict[str, Dict[str, Any]] = {
    "quality": {
        "max_seq_len": 2048,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "num_epochs": 3,
    },
    "balanced": {
        "max_seq_len": 2048,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "num_epochs": 2,
    },
    "economy": {
        "max_seq_len": 1536,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 3e-5,
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
        "num_epochs": 2,
    },
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_env_file(path: str = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip("\"'")  # simple .env format
        if key and key not in os.environ:
            os.environ[key] = val


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open(encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_update(cfg, user_cfg)


def get_gpu_info() -> Tuple[Optional[float], Optional[str]]:
    if not torch.cuda.is_available():
        return None, None
    props = torch.cuda.get_device_properties(0)
    return props.total_memory / 1e9, props.name


def apply_profile(cfg: Dict[str, Any], vram_gb: Optional[float]) -> None:
    profile = cfg.get("profile")
    if profile and profile in PROFILES:
        cfg.update(PROFILES[profile])
        if profile == "quality" and vram_gb and vram_gb >= 40:
            cfg["max_seq_len"] = max(cfg.get("max_seq_len", 2048), 4096)


def auto_configure(cfg: Dict[str, Any], vram_gb: Optional[float]) -> None:
    if platform.system().lower().startswith("win"):
        # bitsandbytes is not supported on Windows in most setups
        cfg["use_4bit"] = False
        cfg["optim"] = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"
        return
    if not torch.cuda.is_available():
        cfg["use_4bit"] = False
        return
    if cfg.get("auto_qlora") and vram_gb is not None:
        cfg["use_4bit"] = vram_gb < float(cfg.get("vram_gb_threshold", 28))


def detect_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except Exception:
        return False


def choose_lora_targets(cfg: Dict[str, Any], train_size: int) -> List[str]:
    attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    full_modules = attn_modules + ["gate_proj", "up_proj", "down_proj"]

    mode = cfg.get("lora_target_modules")
    if isinstance(mode, list) and mode:
        return mode
    if isinstance(mode, str) and mode not in {"auto", ""}:
        if mode == "attention":
            return attn_modules
        if mode == "full":
            return full_modules

    # auto mode
    return full_modules if train_size < 5000 else attn_modules


def pack_sequences(samples: List[Dict[str, Any]], max_seq_len: int) -> List[Dict[str, Any]]:
    if not samples:
        return samples
    packed: List[Dict[str, Any]] = []
    cur_ids: List[int] = []
    cur_labels: List[int] = []

    for s in samples:
        ids = s["input_ids"]
        labels = s["labels"]
        if len(ids) > max_seq_len:
            continue
        if cur_ids and len(cur_ids) + len(ids) > max_seq_len:
            packed.append(
                {
                    "input_ids": cur_ids,
                    "attention_mask": [1] * len(cur_ids),
                    "labels": cur_labels,
                    "length": len(cur_ids),
                }
            )
            cur_ids = []
            cur_labels = []
        cur_ids.extend(ids)
        cur_labels.extend(labels)

    if cur_ids:
        packed.append(
            {
                "input_ids": cur_ids,
                "attention_mask": [1] * len(cur_ids),
                "labels": cur_labels,
                "length": len(cur_ids),
            }
        )
    return packed


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, "w", "utf-8"), logging.StreamHandler()],
    )
    warnings.filterwarnings("ignore", category=FutureWarning)


def notify(url: str, msg: str) -> None:
    if not url:
        return
    try:
        try:
            import requests
            requests.get(url, params={"msg": msg[:1000]}, timeout=5)
        except Exception:
            import urllib.parse
            import urllib.request

            q = urllib.parse.urlencode({"msg": msg[:1000]})
            with urllib.request.urlopen(f"{url}?{q}", timeout=5):
                pass
    except Exception as e:  # noqa: BLE001
        logging.warning("notify failed: %s", e)


def log_env(run_dir: Path) -> None:
    import transformers
    import datasets
    import peft

    env = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "peft": peft.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import bitsandbytes as bnb

        env["bitsandbytes"] = bnb.__version__
    except Exception:
        env["bitsandbytes"] = None
    with (run_dir / "env.json").open("w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def preflight(cfg: Dict[str, Any], run_dir: Path) -> None:
    data_dir = Path(cfg["data_dir"])
    train_path = data_dir / cfg["train_file"]
    val_path = data_dir / cfg["val_file"]
    if not train_path.exists():
        raise FileNotFoundError(f"Train file missing: {train_path}")
    if not val_path.exists():
        logging.warning("Val file missing: %s", val_path)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logging.info("GPU: %s | VRAM: %.1f GB", props.name, props.total_memory / 1e9)
    else:
        logging.warning("CUDA not available; training on CPU will be slow")

    usage = shutil.disk_usage(run_dir)
    logging.info("Disk free: %.1f GB", usage.free / 1e9)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        if cleaned and cleaned[-1]["role"] == role:
            cleaned[-1]["content"] += "\n" + content
        else:
            cleaned.append({"role": role, "content": content})

    if not cleaned:
        return []

    # keep only first system message
    if cleaned[0]["role"] == "system":
        sys_msg = cleaned[0]
        rest = [m for m in cleaned[1:] if m["role"] != "system"]
        cleaned = [sys_msg] + rest
    else:
        cleaned = [m for m in cleaned if m["role"] != "system"]

    # remove leading assistant
    while cleaned and cleaned[0]["role"] == "assistant":
        cleaned.pop(0)

    return cleaned


def clean_text(text: str, max_len: int) -> str:
    if not text:
        return ""
    filtered = "".join(ch if ch.isprintable() else " " for ch in text)
    filtered = " ".join(filtered.split())
    return filtered[:max_len]


def build_prompt(
    tokenizer: AutoTokenizer,
    context: List[Dict[str, str]],
) -> str:
    if not context:
        return ""
    # ensure context ends with user
    while context and context[-1]["role"] != "user":
        context = context[:-1]
    if not context:
        return ""
    return tokenizer.apply_chat_template(
        context,
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_targets(messages: List[Dict[str, str]], mode: str) -> Iterable[Tuple[List[Dict[str, str]], str]]:
    indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not indices:
        return []
    if mode == "last":
        indices = [indices[-1]]
    for idx in indices:
        ctx = messages[:idx]
        ans = messages[idx]["content"].strip()
        if ans:
            yield ctx, ans


def build_samples(
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    tokenizer: AutoTokenizer,
) -> Dataset:
    max_seq_len = int(cfg["max_seq_len"])
    max_answer_tokens = int(cfg.get("max_answer_tokens") or max_seq_len)
    assistant_turns = cfg.get("assistant_turns", "all")
    max_per_dialog = int(cfg.get("max_samples_per_dialog") or 0)

    samples: List[Dict[str, Any]] = []
    eos_id = tokenizer.eos_token_id

    for rec in records:
        msgs = normalize_messages(rec.get("messages", []))
        if len(msgs) < 2:
            continue

        turns = list(iter_targets(msgs, assistant_turns))
        if max_per_dialog and len(turns) > max_per_dialog:
            turns = turns[-max_per_dialog:]

        for ctx, answer in turns:
            ctx = normalize_messages(ctx)
            prompt = build_prompt(tokenizer, ctx)
            if not prompt:
                continue

            prompt_ids = tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=False,
            ).input_ids
            answer_ids = tokenizer(
                answer,
                truncation=True,
                max_length=max_answer_tokens,
                add_special_tokens=False,
            ).input_ids

            max_prompt_len = max_seq_len - len(answer_ids) - 1
            if max_prompt_len <= 0:
                answer_ids = answer_ids[: max_seq_len - 1]
                prompt_ids = []
            elif len(prompt_ids) > max_prompt_len:
                prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + answer_ids + [eos_id]
            if len(input_ids) > max_seq_len:
                continue

            labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]
            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                    "length": len(input_ids),
                }
            )

    if not samples:
        logging.warning("build_samples produced no data")
        return Dataset.from_list([])

    if cfg.get("pack_samples"):
        samples = pack_sequences(samples, max_seq_len)

    ds = Dataset.from_list(samples)
    return ds.with_format("torch")


def build_trainer(
    cfg: Dict[str, Any],
    run_dir: Path,
    model,
    tokenizer,
    train_raw: List[Dict[str, Any]],
    val_raw: List[Dict[str, Any]],
) -> Tuple[Trainer, Dataset, Dataset]:
    train_ds = build_samples(train_raw, cfg, tokenizer)
    val_ds = build_samples(val_raw, cfg, tokenizer) if val_raw else Dataset.from_list([])

    updates = math.ceil(len(train_ds) / cfg["per_device_train_batch_size"] / cfg["gradient_accumulation_steps"]) * cfg["num_epochs"] if len(train_ds) > 0 else 0
    warmup_steps = max(10, int(updates * cfg["warmup_ratio"])) if updates > 0 else 0

    use_bf16 = cfg["use_bf16"]
    if use_bf16 == "auto":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    optim = cfg.get("optim", "auto")
    if optim == "auto":
        optim = "paged_adamw_8bit" if cfg.get("use_4bit") else "adamw_torch_fused"

    eval_strategy = "steps" if len(val_ds) > 0 else "no"
    save_strategy = "steps" if cfg.get("save_steps") else "no"

    params = dict(
        output_dir=str(run_dir),
        run_name=cfg["run_name"],
        seed=int(cfg["seed"]),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=cfg.get("use_fp16", True) and not use_bf16,
        bf16=use_bf16,
        max_grad_norm=cfg["max_grad_norm"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        save_strategy=save_strategy,
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        load_best_model_at_end=bool(len(val_ds)),
        group_by_length=cfg.get("group_by_length", True),
        optim=optim,
        report_to=cfg.get("report_to", []),
        save_safetensors=True,
    )
    try:
        params["eval_strategy"] = eval_strategy
        args = TrainingArguments(**params)
    except TypeError:
        params["evaluation_strategy"] = eval_strategy
        params.pop("eval_strategy", None)
        args = TrainingArguments(**params)

    def collate(batch):
        pad_id = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch], batch_first=True, padding_value=-100
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    callbacks: List[TrainerCallback] = [
        CSVLogger(run_dir / "metrics.csv"),
        SaveAdapterCallback(),
        NotifyLogger(cfg.get("notify_url", ""), cfg.get("notify_interval", 50), cfg),
        EvalPPLCallback(),
        EvalNotifyCallback(cfg.get("notify_url", "")),
    ]
    callbacks.append(
        DatasetSampleNotify(
            cfg.get("notify_url", ""),
            cfg.get("notify_sample_interval", 500),
            train_raw,
            tokenizer,
            cfg,
        )
    )
    if cfg.get("early_stopping_patience") and len(val_ds) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"]))

    if cfg.get("generate_interval", 0) > 0 and cfg.get("sample_prompts"):
        callbacks.append(
            SampleGenerateCallback(
                tokenizer,
                model,
                cfg["sample_prompts"],
                cfg["generate_interval"],
                cfg,
                cfg.get("notify_url", ""),
            )
        )
    if cfg.get("notify_dialog_interval", 0) > 0 and cfg.get("notify_dialogs"):
        callbacks.append(MultiTurnNotifyCallback(tokenizer, cfg, cfg.get("notify_url", "")))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collate,
        callbacks=callbacks,
    )
    return trainer, train_ds, val_ds


def build_model_and_tokenizer(cfg: Dict[str, Any]):
    model_id = cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_bf16 = cfg["use_bf16"]
    if use_bf16 == "auto":
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = cfg.get("use_fp16", True)

    torch_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    quant_cfg = None
    if cfg.get("use_4bit"):
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    attn_impl = None
    if cfg.get("use_flash_attn") in {True, "auto"} and detect_flash_attn():
        attn_impl = "flash_attention_2"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=quant_cfg,
            attn_implementation=attn_impl,
        )
    except Exception as e:
        if attn_impl:
            logging.warning("flash-attn init failed (%s). Falling back to default attention.", e)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                device_map="auto",
                quantization_config=quant_cfg,
            )
        else:
            raise
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            try:
                model.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                model.gradient_checkpointing_enable()
    if cfg.get("use_4bit"):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


class CSVLogger(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path
        self.columns = ["step"]
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        for key in logs:
            if key not in self.columns:
                self.columns.append(key)
        with self.path.open("a", newline="", encoding="utf-8") as f:
            if not self.header_written:
                f.write(",".join(self.columns) + "\n")
                self.header_written = True
            row = {"step": state.global_step}
            row.update({k: logs.get(k, "") for k in self.columns if k != "step"})
            f.write(",".join(str(row.get(k, "")) for k in self.columns) + "\n")


class NotifyLogger(TrainerCallback):
    def __init__(self, url: str, interval: int, cfg: Dict[str, Any]):
        self.url = url
        self.interval = max(1, int(interval or 0))
        self.cfg = cfg
        self.start_time = None
        self.last_time = None
        self.last_step = 0
        self.loss_ema = None
        self.last_loss = None

    def _format_eta(self, seconds: float) -> str:
        if seconds <= 0:
            return "ETA:0s"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"ETA:{h}h{m:02d}m"
        if m:
            return f"ETA:{m}m{s:02d}s"
        return f"ETA:{s}s"

    def _vram_text(self) -> str:
        if not torch.cuda.is_available():
            return "VRAM:cpu"
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        return f"VRAM:{alloc:.1f}/{reserved:.1f}GB peak:{peak:.1f}GB"

    def on_train_begin(self, args, state, control, **kw):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_step = 0

    def on_log(self, args, state, control, logs=None, **kw):
        if not self.url or not logs:
            return
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return

        now = time.time()
        elapsed = (now - self.start_time) if self.start_time else 0.0
        step_delta = state.global_step - self.last_step
        time_delta = (now - self.last_time) if self.last_time else 0.0
        speed = (step_delta / time_delta) if time_delta > 0 else 0.0

        max_steps = state.max_steps or args.max_steps or 0
        if max_steps and elapsed > 0:
            eta_sec = (max_steps - state.global_step) / max(speed, 1e-6)
            eta_text = self._format_eta(eta_sec)
        else:
            eta_text = "ETA:?"

        parts = [
            f"step={state.global_step}/{max_steps if max_steps else '?'}",
            f"speed={speed:.2f} step/s",
            eta_text,
            self._vram_text(),
        ]
        loss = logs.get("loss")
        prev_loss = self.last_loss
        grad_norm = logs.get("grad_norm")
        lr = logs.get("learning_rate")
        epoch = logs.get("epoch", state.epoch)

        if isinstance(loss, (int, float)):
            self.loss_ema = loss if self.loss_ema is None else (0.9 * self.loss_ema + 0.1 * float(loss))
            self.last_loss = float(loss)

        for key in ("loss", "eval_loss", "eval_ppl", "learning_rate", "grad_norm"):
            if key in logs:
                val = logs[key]
                if isinstance(val, float):
                    if key == "learning_rate":
                        parts.append(f"{key}={val:.2e}")
                    elif key == "grad_norm":
                        parts.append(f"{key}={val:.2f}")
                    else:
                        parts.append(f"{key}={val:.4f}")
                else:
                    parts.append(f"{key}={val}")
        if self.loss_ema is not None:
            parts.append(f"loss_ema={self.loss_ema:.4f}")
        if epoch is not None:
            parts.append(f"epoch={epoch:.3f}")

        notify(self.url, " | ".join(parts))
        self._maybe_alerts(loss, grad_norm, prev_loss)
        self.last_time = now
        self.last_step = state.global_step

    def _maybe_alerts(self, loss, grad_norm, prev_loss):
        if not self.url:
            return
        if isinstance(loss, float) and (math.isnan(loss) or math.isinf(loss)):
            notify(self.url, "[ALERT] loss is NaN/Inf")
        if isinstance(grad_norm, float) and (math.isnan(grad_norm) or math.isinf(grad_norm)):
            notify(self.url, "[ALERT] grad_norm is NaN/Inf")
        spike_ratio = float(self.cfg.get("alert_loss_spike_ratio", 1.5))
        grad_thresh = float(self.cfg.get("alert_grad_norm", 10.0))
        if prev_loss and isinstance(loss, (int, float)):
            if loss > prev_loss * spike_ratio:
                notify(self.url, f"[WARN] loss spike: {loss:.4f} > {spike_ratio}x prev")
        if isinstance(grad_norm, (int, float)) and grad_norm > grad_thresh:
            notify(self.url, f"[WARN] grad_norm high: {grad_norm:.2f} > {grad_thresh}")


class EvalPPLCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if not metrics:
            return
        eval_loss = metrics.get("eval_loss")
        prev_eval = None
        if isinstance(eval_loss, (int, float)) and eval_loss > 0:
            metrics["eval_ppl"] = math.exp(float(eval_loss))


class EvalNotifyCallback(TrainerCallback):
    def __init__(self, url: str):
        self.url = url
        self.last_eval_loss = None
        self.worse_streak = 0
        self.best_eval_loss = None

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if not self.url or not metrics:
            return
        parts = [f"[EVAL step {state.global_step}]"]
        eval_loss = metrics.get("eval_loss")
        prev_eval = None
        if isinstance(eval_loss, (int, float)):
            prev_eval = self.last_eval_loss
            if self.best_eval_loss is None or eval_loss < self.best_eval_loss:
                self.best_eval_loss = float(eval_loss)
            if prev_eval is not None and eval_loss > prev_eval:
                self.worse_streak += 1
            else:
                self.worse_streak = 0
            self.last_eval_loss = float(eval_loss)

            if self.worse_streak >= 2:
                notify(self.url, f"[WARN] eval_loss worsens {self.worse_streak + 1} consecutive evals")

        if self.best_eval_loss is not None:
            parts.append(f"best_eval_loss={self.best_eval_loss:.4f}")
        if isinstance(eval_loss, (int, float)) and prev_eval is not None:
            parts.append(f"delta_eval={eval_loss - prev_eval:+.4f}")

        for key in ("eval_loss", "eval_ppl", "learning_rate"):
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    parts.append(f"{key}={val:.4f}" if key != "learning_rate" else f"{key}={val:.2e}")
                else:
                    parts.append(f"{key}={val}")
        if "epoch" in metrics:
            parts.append(f"epoch={metrics['epoch']}")
        notify(self.url, " | ".join(parts))


class DatasetSampleNotify(TrainerCallback):
    def __init__(self, url: str, interval: int, data: List[Dict[str, Any]], tokenizer, cfg: Dict[str, Any]):
        self.url = url
        self.interval = max(1, int(interval or 0))
        self.data = data
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model = None
        self.sent = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.model = kwargs.get("model", None)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.url or self.interval <= 0:
            return
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return
        if not self.data:
            return

        rec = random.choice(self.data)
        msgs = normalize_messages(rec.get("messages", []))
        if len(msgs) < 2:
            return
        # pick last assistant as target
        targets = list(iter_targets(msgs, "last"))
        if not targets:
            return
        ctx, gold = targets[-1]
        prompt = build_prompt(self.tokenizer, normalize_messages(ctx))
        if not prompt:
            return

        model = kwargs.get("model") or self.model
        if model is None:
            return

        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg["max_seq_len"])
        enc = {k: v.to(model.device) for k, v in enc.items()}
        do_sample = bool(self.cfg.get("notify_sample_do_sample", False))
        gen_kwargs = {
            "max_new_tokens": self.cfg.get("max_gen_tokens", 128),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": self.cfg.get("repetition_penalty", 1.1),
            "no_repeat_ngram_size": self.cfg.get("no_repeat_ngram_size", 3),
        }
        if do_sample:
            gen_kwargs.update(
                {
                    "temperature": self.cfg.get("temperature", 0.7),
                    "top_p": self.cfg.get("top_p", 0.9),
                }
            )
        was_training = model.training
        model.eval()
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        if was_training:
            model.train()
        gen = self.tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
        gen = clean_text(gen, 800)
        gold = clean_text(gold, 800)
        prompt = clean_text(prompt, 1200)
        self.sent += 1
        msg = f"[SAMPLE #{self.sent} step {state.global_step}]\\nGOLD: {gold}\\nGEN: {gen}\\nPROMPT: {prompt}"
        notify(self.url, msg)


class SaveAdapterCallback(TrainerCallback):
    def on_save(self, args, state, control, **kw):
        model = kw.get("model")
        if not model:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "adapter"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        logging.info("Adapter saved: %s", ckpt_dir)


class SampleGenerateCallback(TrainerCallback):
    def __init__(self, tokenizer, model, samples: List[str], interval: int, cfg: Dict[str, Any], notify_url: str):
        self.tokenizer = tokenizer
        self.model = model
        self.samples = samples
        self.interval = interval
        self.cfg = cfg
        self.notify_url = notify_url

    def _generate(self, prompt: str) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg["max_seq_len"])
        input_ids = enc.input_ids.to(self.model.device)
        attn_mask = enc.attention_mask.to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.cfg["max_gen_tokens"],
                temperature=self.cfg["temperature"],
                top_p=self.cfg["top_p"],
                repetition_penalty=self.cfg["repetition_penalty"],
                no_repeat_ngram_size=self.cfg["no_repeat_ngram_size"],
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval <= 0 or state.global_step == 0 or state.global_step % self.interval != 0:
            return
        prompt = random.choice(self.samples) if self.samples else None
        if not prompt:
            return
        gen = self._generate(prompt)
        msg = f"[GEN step {state.global_step}]\\nPROMPT: {prompt}\\nGEN: {gen}"
        logging.info(msg)
        notify(self.notify_url, msg)


class MultiTurnNotifyCallback(TrainerCallback):
    def __init__(self, tokenizer, cfg: Dict[str, Any], notify_url: str):
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.notify_url = notify_url
        self.dialogs = cfg.get("notify_dialogs") or []
        self.interval = int(cfg.get("notify_dialog_interval") or 0)
        self.turns = int(cfg.get("notify_dialog_turns") or 4)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.notify_url or self.interval <= 0:
            return
        if state.global_step == 0 or state.global_step % self.interval != 0:
            return
        if not self.dialogs:
            return
        model = kwargs.get("model")
        if model is None:
            return

        dialog = random.choice(self.dialogs)
        history: List[Dict[str, str]] = []
        transcript: List[str] = [f"[DIALOG step {state.global_step}]"]

        was_training = model.training
        model.eval()
        for turn_idx, turn in enumerate(dialog[: self.turns]):
            role = (turn.get("role") or "user").strip()
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                history.append({"role": "assistant", "content": content})
                transcript.append(f"A: {clean_text(content, 400)}")
                continue

            history.append({"role": "user", "content": content})
            prompt = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg["max_seq_len"])
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen_kwargs = {
                "max_new_tokens": self.cfg.get("max_gen_tokens", 128),
                "do_sample": bool(self.cfg.get("notify_sample_do_sample", False)),
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": self.cfg.get("repetition_penalty", 1.1),
                "no_repeat_ngram_size": self.cfg.get("no_repeat_ngram_size", 3),
            }
            if gen_kwargs["do_sample"]:
                gen_kwargs.update(
                    {
                        "temperature": self.cfg.get("temperature", 0.7),
                        "top_p": self.cfg.get("top_p", 0.9),
                    }
                )
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            gen = self.tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            gen = clean_text(gen, 500)
            transcript.append(f"U: {clean_text(content, 400)}")
            transcript.append(f"A*: {gen}")
            history.append({"role": "assistant", "content": gen})

        if was_training:
            model.train()
        notify(self.notify_url, "\n".join(transcript))


def run_sanity_checks(
    model,
    tokenizer,
    dialogs: List[List[Dict[str, str]]],
    cfg: Dict[str, Any],
    run_dir: Path,
) -> None:
    results = []
    for idx, dialog in enumerate(dialogs):
        history: List[Dict[str, str]] = []
        dialog_results = {"dialog_index": idx, "turns": []}
        for turn in dialog:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role == "assistant":
                history.append({"role": "assistant", "content": content})
                dialog_results["turns"].append({"role": "assistant", "content": content, "type": "gold"})
                continue

            history.append({"role": "user", "content": content})
            prompt = tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg["max_seq_len"])
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=cfg.get("max_gen_tokens", 128),
                    do_sample=True,
                    temperature=cfg.get("temperature", 0.7),
                    top_p=cfg.get("top_p", 0.9),
                    repetition_penalty=cfg.get("repetition_penalty", 1.1),
                    no_repeat_ngram_size=cfg.get("no_repeat_ngram_size", 3),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen = tokenizer.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
            history.append({"role": "assistant", "content": gen})
            dialog_results["turns"].append({"role": "user", "content": content, "type": "prompt"})
            dialog_results["turns"].append({"role": "assistant", "content": gen, "type": "gen"})

        results.append(dialog_results)

    if results:
        with (run_dir / "sanity_checks.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for Mistral")
    parser.add_argument("--config", help="Path to JSON config", default=None)
    args = parser.parse_args()

    load_env_file(".env")
    if args.config is None:
        default_cfg = Path("config/train_config.json")
        if default_cfg.exists():
            args.config = str(default_cfg)

    cfg = load_config(args.config)
    env_notify = os.environ.get("NOTIFY_URL", "").strip()
    if env_notify and not cfg.get("notify_url"):
        cfg["notify_url"] = env_notify
    vram_gb, gpu_name = get_gpu_info()
    apply_profile(cfg, vram_gb)
    auto_configure(cfg, vram_gb)
    seed_everything(int(cfg["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run_name"]
    if cfg.get("run_dir"):
        run_dir = Path(cfg["run_dir"])
    else:
        run_dir = Path(cfg["output_dir"]) / f"{run_name}_{timestamp}"
    setup_logging(run_dir)
    preflight(cfg, run_dir)
    if vram_gb:
        logging.info("GPU detected: %s (%.1f GB)", gpu_name, vram_gb)
    logging.info("Profile: %s | use_4bit=%s | max_seq_len=%s", cfg.get("profile"), cfg.get("use_4bit"), cfg.get("max_seq_len"))
    logging.info(
        "Training params: save_steps=%s | eval_steps=%s | logging_steps=%s | notify_interval=%s | notify_sample_interval=%s | notify_dialog_interval=%s",
        cfg.get("save_steps"),
        cfg.get("eval_steps"),
        cfg.get("logging_steps"),
        cfg.get("notify_interval"),
        cfg.get("notify_sample_interval"),
        cfg.get("notify_dialog_interval"),
    )
    logging.info(
        "Model params: lr=%s | batch=%s | grad_accum=%s | max_seq_len=%s | lora_targets=%s | pack=%s",
        cfg.get("learning_rate"),
        cfg.get("per_device_train_batch_size"),
        cfg.get("gradient_accumulation_steps"),
        cfg.get("max_seq_len"),
        cfg.get("lora_target_modules"),
        cfg.get("pack_samples"),
    )
    if cfg.get("notify_url"):
        logging.info("Notify enabled: %s", cfg["notify_url"])
    else:
        logging.warning("Notify URL not set; notifications will be skipped")

    with (run_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log_env(run_dir)

    train_raw = read_jsonl(Path(cfg["data_dir"]) / cfg["train_file"])
    val_raw = read_jsonl(Path(cfg["data_dir"]) / cfg["val_file"])

    cfg["lora_target_modules"] = choose_lora_targets(cfg, len(train_raw))
    logging.info("LoRA target modules: %s", cfg["lora_target_modules"])

    model, tokenizer = build_model_and_tokenizer(cfg)

    ckpt = None
    if cfg.get("resume"):
        ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        ckpt = str(ckpts[-1]) if ckpts else None
        if ckpt and not cfg.get("resume_optimizer", True):
            for fname in ("optimizer.pt", "scheduler.pt"):
                fpath = Path(ckpt) / fname
                if fpath.exists():
                    fpath.unlink()
            logging.warning("Optimizer/scheduler state removed for resume: %s", ckpt)

    retries = int(cfg.get("oom_retries", 0))
    for attempt in range(retries + 1):
        trainer, train_ds, val_ds = build_trainer(cfg, run_dir, model, tokenizer, train_raw, val_raw)
        logging.info("Samples: %d train | %d val", len(train_ds), len(val_ds))
        notify(
            cfg.get("notify_url", ""),
            (
                f"Train start: {run_dir.name} (resume={bool(ckpt)}) | "
                f"save_steps={cfg.get('save_steps')} | eval_steps={cfg.get('eval_steps')} | "
                f"max_seq_len={cfg.get('max_seq_len')} | lora={cfg.get('lora_target_modules')}"
            ),
        )
        try:
            trainer.train(resume_from_checkpoint=ckpt)
            notify(cfg.get("notify_url", ""), "Train finished")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and cfg.get("auto_oom_recovery") and attempt < retries:
                logging.warning("OOM detected, attempting recovery: %s", e)
                notify(cfg.get("notify_url", ""), "OOM detected, auto-recovery adjusting max_seq_len/grad_accum")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                cfg["max_seq_len"] = max(1024, int(cfg["max_seq_len"]) - 512)
                cfg["gradient_accumulation_steps"] = min(int(cfg["gradient_accumulation_steps"]) * 2, 64)
                logging.info(
                    "Adjusted max_seq_len=%s, gradient_accumulation_steps=%s",
                    cfg["max_seq_len"],
                    cfg["gradient_accumulation_steps"],
                )
                ckpt = None
                continue
            raise

    adapter_dir = run_dir / "final_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logging.info("Adapter saved to %s", adapter_dir)

    if cfg.get("sanity_dialogs"):
        run_sanity_checks(model, tokenizer, cfg["sanity_dialogs"], cfg, run_dir)

    if cfg.get("save_merged") and not cfg.get("use_4bit"):
        merged_dir = run_dir / "merged_model"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logging.info("Merged model saved to %s", merged_dir)


if __name__ == "__main__":
    main()
