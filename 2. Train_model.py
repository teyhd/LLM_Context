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

# ========================
# Default config
# ========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
    "data_dir": "data/output",
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    "output_dir": "models",
    "run_name": "vlad",
    "run_dir": "",
    "resume": True,
    "seed": 42,
    "num_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_seq_len": 2048,
    "max_answer_tokens": 1024,
    "max_grad_norm": 0.3,
    "use_fp16": True,
    "use_bf16": "auto",
    "gradient_checkpointing": True,
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "save_steps": 200,
    "eval_steps": 200,
    "save_total_limit": 4,
    "logging_steps": 10,
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
    "assistant_turns": "all",  # all | last
    "max_samples_per_dialog": 0,
    "notify_url": "",
    "save_merged": False,
}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


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
        import requests

        requests.get(url, params={"msg": msg[:1000]})
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

    ds = Dataset.from_list(samples)
    return ds.with_format("torch")


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
        compute_dtype = torch.float16 if cfg.get("bnb_4bit_compute_dtype") == "float16" else torch.bfloat16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_cfg,
    )
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing"):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for Mistral")
    parser.add_argument("--config", help="Path to JSON config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg["run_name"]
    if cfg.get("run_dir"):
        run_dir = Path(cfg["run_dir"])
    else:
        run_dir = Path(cfg["output_dir"]) / f"{run_name}_{timestamp}"
    setup_logging(run_dir)
    preflight(cfg, run_dir)

    with (run_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    log_env(run_dir)

    train_raw = read_jsonl(Path(cfg["data_dir"]) / cfg["train_file"])
    val_raw = read_jsonl(Path(cfg["data_dir"]) / cfg["val_file"])

    model, tokenizer = build_model_and_tokenizer(cfg)
    train_ds = build_samples(train_raw, cfg, tokenizer)
    val_ds = build_samples(val_raw, cfg, tokenizer) if val_raw else Dataset.from_list([])

    logging.info("Samples: %d train | %d val", len(train_ds), len(val_ds))

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

    args = TrainingArguments(
        output_dir=str(run_dir),
        run_name=run_name,
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
        evaluation_strategy=eval_strategy,
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
    ]
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        data_collator=collate,
        callbacks=callbacks,
    )

    ckpt = None
    if cfg.get("resume"):
        ckpts = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        ckpt = str(ckpts[-1]) if ckpts else None
    notify(cfg.get("notify_url", ""), f"Train start: {run_dir.name} (resume={bool(ckpt)})")
    trainer.train(resume_from_checkpoint=ckpt)
    notify(cfg.get("notify_url", ""), "Train finished")

    adapter_dir = run_dir / "final_adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logging.info("Adapter saved to %s", adapter_dir)

    if cfg.get("save_merged") and not cfg.get("use_4bit"):
        merged_dir = run_dir / "merged_model"
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        logging.info("Merged model saved to %s", merged_dir)


if __name__ == "__main__":
    main()
