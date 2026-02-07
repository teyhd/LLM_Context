# LLM_CONTEXT – Mistral SFT on Telegram dialogs

This repo contains:
- `1. Dataset_Builder.py` – builds/cleans a JSONL chat dataset from Telegram exports
- `2. Train_model.py` – trains a LoRA/QLoRA adapter for Mistral
- `config/` – JSON configs for dataset building and training

## Quick start

1) Build dataset + report
```
python "1. Dataset_Builder.py" --config config/dataset_config.json
```

Artifacts:
- `data/output/train.jsonl`, `data/output/val.jsonl`
- `data/output/report.json` (dataset stats + exclusions)
- `data/output/samples.jsonl` (N random samples if `dry_run > 0`)
- `data/output/excluded.jsonl` (dropped messages)

2) Train LoRA/QLoRA
```
python "2. Train_model.py" --config config/train_config.json
```

Artifacts (per run):
- `models/vlad_YYYYMMDD_HHMMSS/final_adapter`
- `models/vlad_YYYYMMDD_HHMMSS/metrics.csv`
- `models/vlad_YYYYMMDD_HHMMSS/train.log`
- `models/vlad_YYYYMMDD_HHMMSS/env.json`
- `models/vlad_YYYYMMDD_HHMMSS/train_config.json`

Resume: set `"run_dir": "models/vlad_YYYYMMDD_HHMMSS"` in the config and re-run.

## Inference (adapter)

Use `scripts/infer_adapter.py` (see below) after training:
```
python scripts/infer_adapter.py --base mistralai/Mistral-7B-Instruct-v0.3 --adapter models/vlad_YYYYMMDD_HHMMSS/final_adapter
```

## Recommended hyperparameters (Tesla P40, 24GB)

Default profile (config/train_config.json):
- QLoRA 4-bit, `max_seq_len=2048`
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=4`
- `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.1`
- `learning_rate=2e-5`, `warmup_ratio=0.05`

Economy profile (for tighter VRAM / more stability):
- `max_seq_len=1536`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
- `lora_r=8`, `lora_alpha=16`
- `learning_rate=1.5e-5`

## Notes

- Dataset builder masks URLs/emails/phones by default and can drop PII if needed.
- Training uses all assistant turns by default (`assistant_turns=all`).
- If you want to generate samples during training, set `generate_interval > 0` and provide `sample_prompts`.
