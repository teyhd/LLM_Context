import argparse
import json
import logging
import os
import random
import re
import hashlib
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ========================
# Default config (override via --config)
# ========================

DEFAULT_CONFIG: Dict[str, Any] = {
    "input_dir": "data/input",
    "input_glob": "*.json",
    "output_dir": "data/output",
    "train_file": "train.jsonl",
    "val_file": "val.jsonl",
    "report_file": "report.json",
    "samples_file": "samples.jsonl",
    "excluded_file": "excluded.jsonl",
    "log_file": "build_dataset.log",
    "system_prompt": (
        "Ты Влад. Ты дружелюбный и лаконичный парень.\n"
        "Главный фокус — переписка: отвечай по делу, без лишней воды."
    ),
    "assistant_names": ["Vs"],
    "assistant_ids": ["user304622290"],
    "time_gap_hours": 2.0,
    "merge_same_role_minutes": 10,
    "reply_max_hours": 24.0,
    "reply_max_messages": 200,
    "min_messages_in_block": 2,
    "max_messages_in_block": 80,
    "max_message_chars": 2000,
    "max_block_chars": 12000,
    "min_assistant_chars": 5,
    "keep_short_assistants": True,
    "short_assistant_keep_prob": 0.35,
    "short_phrases": [
        "ок",
        "ok",
        "да",
        "угу",
        "ага",
        "понял",
        "принял",
        "нет",
        "спасибо",
        "хорошо",
        "ясно",
    ],
    "keep_forwarded": False,
    "keep_media_as_tokens": False,
    "media_token": "<MEDIA>",
    "pii_policy": "mask",  # mask | drop
    "cleaning": {
        "strip_urls": True,
        "strip_emails": True,
        "strip_phones": True,
        "strip_long_tokens": True,
        "strip_extra_whitespace": True,
    },
    "dedup": {
        "enabled": True,
        "simhash_bits": 64,
        "simhash_distance": 2,
        "bucket_bits": 12,
        "min_tokens": 8,
    },
    "split": {
        "val_ratio": 0.02,
        "seed": 42,
        "by_dialog": True,
    },
    "dry_run": 0,
}

# ========================
# Regex helpers
# ========================

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-\(\)]{7,}\d)")
LONG_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{32,}\b")
TOKEN_SPLIT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open(encoding="utf-8") as f:
        user_cfg = json.load(f)
    return deep_update(cfg, user_cfg)


def setup_logging(output_dir: Path, log_file: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / log_file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, "w", "utf-8"), logging.StreamHandler()],
    )


def read_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except Exception:
            continue
    raise ValueError(f"Failed to decode JSON: {path}")


def parse_datetime(msg: Dict[str, Any]) -> Optional[datetime]:
    if msg.get("date_unixtime"):
        try:
            return datetime.utcfromtimestamp(int(msg["date_unixtime"]))
        except Exception:
            pass
    if msg.get("date"):
        try:
            return datetime.fromisoformat(msg["date"].replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def flatten_text(text: Any) -> str:
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: List[str] = []
        for item in text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def normalize_whitespace(text: str) -> str:
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def clean_text(text: str, cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    flags: List[str] = []
    t = text
    if cfg["cleaning"].get("strip_urls"):
        t2 = URL_RE.sub("<URL>", t)
        if t2 != t:
            flags.append("url")
        t = t2
    if cfg["cleaning"].get("strip_emails"):
        t2 = EMAIL_RE.sub("<EMAIL>", t)
        if t2 != t:
            flags.append("email")
        t = t2
    if cfg["cleaning"].get("strip_phones"):
        t2 = PHONE_RE.sub("<PHONE>", t)
        if t2 != t:
            flags.append("phone")
        t = t2
    if cfg["cleaning"].get("strip_long_tokens"):
        t2 = LONG_TOKEN_RE.sub("<TOKEN>", t)
        if t2 != t:
            flags.append("token")
        t = t2
    if cfg["cleaning"].get("strip_extra_whitespace"):
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(), flags


def estimate_tokens(text: str) -> int:
    return len(TOKEN_SPLIT_RE.findall(text))


def is_assistant(msg: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    return msg.get("from") in cfg["assistant_names"] or msg.get("from_id") in cfg["assistant_ids"]


def summarize_distribution(values: List[int]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p95": None, "p99": None}
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    def pct(p: float) -> int:
        return vals[min(n - 1, max(0, int(p * n) - 1))]
    return {
        "count": n,
        "min": vals[0],
        "max": vals[-1],
        "mean": round(mean, 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


def ratio(a: int, b: int) -> Optional[float]:
    if b == 0:
        return None
    return round(a / b, 4)


def is_noise_duplicate_text(text: str) -> bool:
    if not text or len(text) < 3:
        return True
    if text in {"<url>", "<email>", "<phone>", "<token>"}:
        return True
    if not re.search(r"[A-Za-zА-Яа-я0-9]", text):
        return True
    return False


def compute_final_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    assistant_len_tokens: List[int] = []
    assistant_len_chars: List[int] = []
    block_len_tokens: List[int] = []
    block_len_chars: List[int] = []
    block_len_messages: List[int] = []
    duplicate_counter: Counter = Counter()

    stats = {
        "assistant_messages": 0,
        "user_messages": 0,
        "assistant_tokens": 0,
        "user_tokens": 0,
        "assistant_chars": 0,
        "user_chars": 0,
    }

    for rec in records:
        msgs = rec.get("messages", [])
        block_tokens = 0
        block_chars = 0
        block_msgs = 0
        last_ass = None
        for m in msgs:
            if m.get("role") == "system":
                continue
            content = m.get("content", "")
            if not content:
                continue
            block_msgs += 1
            t = estimate_tokens(content)
            c = len(content)
            block_tokens += t
            block_chars += c
            if m["role"] == "assistant":
                stats["assistant_messages"] += 1
                stats["assistant_tokens"] += t
                stats["assistant_chars"] += c
                assistant_len_tokens.append(t)
                assistant_len_chars.append(c)
                last_ass = content
            elif m["role"] == "user":
                stats["user_messages"] += 1
                stats["user_tokens"] += t
                stats["user_chars"] += c

        block_len_tokens.append(block_tokens)
        block_len_chars.append(block_chars)
        block_len_messages.append(block_msgs)

        if last_ass:
            norm_last = normalize_for_dedup(
                [{"role": "user", "content": ""}, {"role": "assistant", "content": last_ass}]
            )
            if norm_last and not is_noise_duplicate_text(norm_last):
                duplicate_counter[norm_last] += 1

    return {
        "stats": stats,
        "length_stats": {
            "assistant_tokens": summarize_distribution(assistant_len_tokens),
            "assistant_chars": summarize_distribution(assistant_len_chars),
            "block_tokens": summarize_distribution(block_len_tokens),
            "block_chars": summarize_distribution(block_len_chars),
            "block_messages": summarize_distribution(block_len_messages),
        },
        "ratios": {
            "assistant_vs_user_messages": ratio(stats["assistant_messages"], stats["user_messages"]),
            "assistant_vs_user_tokens": ratio(stats["assistant_tokens"], stats["user_tokens"]),
            "short_assistant_ratio": None,
        },
        "top_duplicates": [
            {"count": c, "text": t[:200]}
            for t, c in duplicate_counter.most_common(20)
            if c > 1
        ],
    }


def should_drop_pii(flags: List[str], cfg: Dict[str, Any]) -> bool:
    if cfg.get("pii_policy") != "drop":
        return False
    return bool(flags)


def parse_message(
    msg: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], List[str]]:
    if msg.get("type") != "message":
        return None, "non_message_type", []
    if msg.get("action"):
        return None, "service_action", []
    if (not cfg.get("keep_forwarded")) and (
        msg.get("forwarded_from") or msg.get("forwarded_from_chat")
    ):
        return None, "forwarded", []

    dt = parse_datetime(msg)
    if not dt:
        return None, "bad_datetime", []

    raw_text = flatten_text(msg.get("text"))
    raw_text = normalize_whitespace(raw_text)

    if not raw_text.strip():
        if cfg.get("keep_media_as_tokens") and (
            msg.get("media_type") or msg.get("photo") or msg.get("file") or msg.get("sticker_emoji")
        ):
            raw_text = cfg.get("media_token", "<MEDIA>")
        else:
            return None, "empty_text", []

    cleaned, flags = clean_text(raw_text, cfg)
    if not cleaned:
        return None, "empty_after_clean", flags
    if should_drop_pii(flags, cfg):
        return None, "pii_drop", flags

    if cfg.get("max_message_chars") and len(cleaned) > cfg["max_message_chars"]:
        return None, "too_long_message", flags

    role = "assistant" if is_assistant(msg, cfg) else "user"
    parsed = {
        "role": role,
        "content": cleaned,
        "time": dt,
        "id": msg.get("id"),
        "from": msg.get("from"),
        "from_id": msg.get("from_id"),
        "reply_to": msg.get("reply_to_message_id"),
    }
    return parsed, None, flags


def split_into_blocks(messages: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    if not messages:
        return []

    messages.sort(key=lambda m: m["time"])
    blocks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_time: Optional[datetime] = None
    recent_ids: deque = deque(maxlen=int(cfg.get("reply_max_messages") or 0) or 0)

    gap_limit = timedelta(hours=cfg.get("time_gap_hours", 2.0))
    reply_limit = timedelta(hours=cfg.get("reply_max_hours", 24.0))

    for m in messages:
        if not current:
            current = [m]
            last_time = m["time"]
            recent_ids.clear()
            if m.get("id") is not None:
                recent_ids.append(m["id"])
            continue

        gap = m["time"] - (last_time or m["time"])
        reply_bridge = False
        if m.get("reply_to") and recent_ids:
            if m["reply_to"] in recent_ids and gap <= reply_limit:
                reply_bridge = True

        if gap > gap_limit and not reply_bridge:
            blocks.append(current)
            current = [m]
            recent_ids.clear()
        else:
            current.append(m)

        last_time = m["time"]
        if m.get("id") is not None and recent_ids.maxlen:
            recent_ids.append(m["id"])

    if current:
        blocks.append(current)
    return blocks


def compact_block(block: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Dict[str, Any]]:
    meta = {"tail_trimmed": False, "assistant_merges": 0}
    if not block:
        return None, "empty_block", meta

    merge_window = timedelta(minutes=cfg.get("merge_same_role_minutes", 10))
    merged: List[Dict[str, Any]] = []
    for m in block:
        if not merged:
            merged.append(m.copy())
            continue

        prev = merged[-1]
        same_role = prev["role"] == m["role"]
        same_speaker = prev.get("from") == m.get("from") if m["role"] == "user" else True
        if same_role and same_speaker and (m["time"] - prev["time"] <= merge_window):
            prev["content"] += "\n" + m["content"]
            prev["time"] = m["time"]
            if prev["role"] == "assistant":
                meta["assistant_merges"] += 1
        else:
            merged.append(m.copy())

    if not any(m["role"] == "assistant" for m in merged):
        return None, "no_assistant", meta
    if not any(m["role"] == "user" for m in merged):
        return None, "no_user", meta

    # Trim to last assistant
    last_ass_idx = max(i for i, m in enumerate(merged) if m["role"] == "assistant")
    if last_ass_idx < len(merged) - 1:
        meta["tail_trimmed"] = True
    merged = merged[: last_ass_idx + 1]

    # Remove leading assistant (must start with user)
    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)

    if len(merged) < cfg.get("min_messages_in_block", 2):
        return None, "too_few_messages", meta

    # Enforce max sizes by trimming oldest messages
    max_msgs = cfg.get("max_messages_in_block")
    if max_msgs and len(merged) > max_msgs:
        merged = merged[-max_msgs:]
        while merged and merged[0]["role"] == "assistant":
            merged.pop(0)

    max_chars = cfg.get("max_block_chars")
    if max_chars:
        while merged and sum(len(m["content"]) for m in merged) > max_chars:
            merged.pop(0)
        while merged and merged[0]["role"] == "assistant":
            merged.pop(0)

    if not merged:
        return None, "empty_after_trim", meta
    if not any(m["role"] == "assistant" for m in merged):
        return None, "no_assistant_after_trim", meta
    if not any(m["role"] == "user" for m in merged):
        return None, "no_user_after_trim", meta

    return merged, None, meta


def add_speaker_prefix(messages: List[Dict[str, Any]]) -> None:
    user_names = {m.get("from") for m in messages if m["role"] == "user" and m.get("from")}
    if len(user_names) <= 1:
        return
    for m in messages:
        if m["role"] == "user" and m.get("from"):
            m["content"] = f"[{m['from']}] {m['content']}"


def is_short_assistant(text: str, cfg: Dict[str, Any]) -> bool:
    if len(text.strip()) < cfg.get("min_assistant_chars", 5):
        return True
    norm = text.strip().lower()
    return norm in cfg.get("short_phrases", [])


def normalize_for_dedup(messages: List[Dict[str, Any]]) -> str:
    # Focus dedup on the final turn (last user + last assistant) to avoid
    # over-pruning long dialogs with shared templates.
    last_ass = None
    last_user = None
    for m in reversed(messages):
        if m["role"] == "assistant" and not last_ass:
            last_ass = m["content"]
        elif m["role"] == "user" and not last_user:
            last_user = m["content"]
        if last_ass and last_user:
            break

    parts = [p for p in [last_user, last_ass] if p]
    text = "\n".join(parts)
    text = text.lower()
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = PHONE_RE.sub("<PHONE>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simhash(text: str, bits: int = 64) -> int:
    tokens = TOKEN_SPLIT_RE.findall(text.lower())
    if not tokens:
        return 0
    v = [0] * bits
    for t in tokens:
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] >= 0:
            out |= (1 << i)
    return out


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def deduplicate(
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    stats: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not cfg.get("dedup", {}).get("enabled"):
        return records

    dedup_cfg = cfg["dedup"]
    bits = int(dedup_cfg.get("simhash_bits", 64))
    distance = int(dedup_cfg.get("simhash_distance", 2))
    bucket_bits = int(dedup_cfg.get("bucket_bits", 12))
    min_tokens = int(dedup_cfg.get("min_tokens", 20))
    bucket_shift = max(bits - bucket_bits, 0)

    exact_seen: Dict[str, int] = {}
    buckets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    kept: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        norm = normalize_for_dedup(rec["messages"])
        if not norm:
            stats["dedup_exact_removed"] += 1
            continue
        if norm in exact_seen:
            stats["dedup_exact_removed"] += 1
            stats["duplicates"].append(
                {"type": "exact", "first_index": exact_seen[norm], "dup_index": idx}
            )
            continue

        tokens = TOKEN_SPLIT_RE.findall(norm)
        if len(tokens) >= min_tokens:
            sh = simhash(norm, bits=bits)
            bucket = sh >> bucket_shift if bucket_shift else 0
            is_near = False
            for existing_idx, existing_hash in buckets.get(bucket, []):
                if hamming_distance(sh, existing_hash) <= distance:
                    is_near = True
                    stats["dedup_near_removed"] += 1
                    stats["duplicates"].append(
                        {"type": "near", "first_index": existing_idx, "dup_index": idx}
                    )
                    break
            if is_near:
                continue
            buckets[bucket].append((idx, sh))

        exact_seen[norm] = idx
        kept.append(rec)

    return kept


def split_train_val(
    records: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_cfg = cfg["split"]
    ratio = float(split_cfg.get("val_ratio", 0.02))
    seed = int(split_cfg.get("seed", 42))
    by_dialog = split_cfg.get("by_dialog", True)
    rng = random.Random(seed)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []

    if not by_dialog:
        for rec in records:
            (val if rng.random() < ratio else train).append(rec)
        return train, val

    for rec in records:
        dialog_id = rec.get("meta", {}).get("dialog_id") or ""
        h = hashlib.md5(dialog_id.encode("utf-8")).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        (val if bucket < ratio else train).append(rec)
    return train, val


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_records(records: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for i, rec in enumerate(records):
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"record {i}: empty messages")
            continue
        for m in msgs:
            if m.get("role") not in {"system", "user", "assistant"}:
                errors.append(f"record {i}: bad role {m.get('role')}")
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                errors.append(f"record {i}: empty content")
    return errors


def build_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    assistant_len_tokens: List[int] = []
    assistant_len_chars: List[int] = []
    block_len_tokens: List[int] = []
    block_len_chars: List[int] = []
    block_len_messages: List[int] = []
    duplicate_counter: Counter = Counter()

    stats = {
        "messages_total": 0,
        "messages_kept": 0,
        "messages_dropped": 0,
        "blocks_total": 0,
        "blocks_kept": 0,
        "blocks_dropped": 0,
        "tail_trimmed_blocks": 0,
        "assistant_merges": 0,
        "short_assistant_total": 0,
        "short_assistant_kept": 0,
        "dedup_exact_removed": 0,
        "dedup_near_removed": 0,
        "assistant_messages": 0,
        "user_messages": 0,
        "assistant_tokens": 0,
        "user_tokens": 0,
        "assistant_chars": 0,
        "user_chars": 0,
        "reasons": Counter(),
        "per_person": {},
        "duplicates": [],
        "examples_excluded": [],
    }

    records: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []

    input_dir = Path(cfg["input_dir"])
    pattern = cfg["input_glob"]
    files = sorted(input_dir.glob(pattern))
    logging.info("Found %d input files (pattern %s)", len(files), input_dir / pattern)

    for path in files:
        data = read_json(path)
        name = data.get("name") or path.stem
        pstats = stats["per_person"].setdefault(
            name,
            {
                "messages_total": 0,
                "messages_kept": 0,
                "messages_dropped": 0,
                "blocks_total": 0,
                "blocks_kept": 0,
                "blocks_dropped": 0,
            },
        )

        prepared: List[Dict[str, Any]] = []
        for msg in data.get("messages", []):
            stats["messages_total"] += 1
            pstats["messages_total"] += 1
            parsed, reason, flags = parse_message(msg, cfg)
            if not parsed:
                stats["messages_dropped"] += 1
                pstats["messages_dropped"] += 1
                stats["reasons"][reason] += 1
                if len(stats["examples_excluded"]) < 50:
                    stats["examples_excluded"].append(
                        {
                            "file": path.name,
                            "id": msg.get("id"),
                            "from": msg.get("from"),
                            "reason": reason,
                            "flags": flags,
                            "text": flatten_text(msg.get("text"))[:200],
                        }
                    )
                excluded_rows.append(
                    {
                        "file": path.name,
                        "id": msg.get("id"),
                        "from": msg.get("from"),
                        "reason": reason,
                        "flags": flags,
                        "text": flatten_text(msg.get("text")),
                    }
                )
                continue

            stats["messages_kept"] += 1
            pstats["messages_kept"] += 1
            prepared.append(parsed)

        blocks = split_into_blocks(prepared, cfg)
        logging.info(
            "Processed %s (%s): %d messages -> %d blocks",
            path.name,
            name,
            len(prepared),
            len(blocks),
        )

        for block in blocks:
            stats["blocks_total"] += 1
            pstats["blocks_total"] += 1

            compact, reason, meta = compact_block(block, cfg)
            stats["assistant_merges"] += meta["assistant_merges"]
            if meta["tail_trimmed"]:
                stats["tail_trimmed_blocks"] += 1

            if compact is None:
                stats["blocks_dropped"] += 1
                pstats["blocks_dropped"] += 1
                stats["reasons"][reason] += 1
                excluded_rows.append(
                    {
                        "file": path.name,
                        "reason": reason,
                        "block_preview": [
                            {"role": m["role"], "content": m["content"][:200]}
                            for m in block[:5]
                        ],
                    }
                )
                continue

            add_speaker_prefix(compact)

            last_ass = [m for m in compact if m["role"] == "assistant"][-1]
            if is_short_assistant(last_ass["content"], cfg):
                stats["short_assistant_total"] += 1
                keep = cfg.get("keep_short_assistants", True)
                if keep and random.random() <= cfg.get("short_assistant_keep_prob", 0.35):
                    stats["short_assistant_kept"] += 1
                elif keep:
                    stats["blocks_dropped"] += 1
                    pstats["blocks_dropped"] += 1
                    stats["reasons"]["short_assistant_downsampled"] += 1
                    continue
                else:
                    stats["blocks_dropped"] += 1
                    pstats["blocks_dropped"] += 1
                    stats["reasons"]["short_assistant_dropped"] += 1
                    continue

            stats["blocks_kept"] += 1
            pstats["blocks_kept"] += 1

            for m in compact:
                if m["role"] == "assistant":
                    stats["assistant_messages"] += 1
                    t = estimate_tokens(m["content"])
                    c = len(m["content"])
                    stats["assistant_tokens"] += t
                    stats["assistant_chars"] += c
                elif m["role"] == "user":
                    stats["user_messages"] += 1
                    stats["user_tokens"] += estimate_tokens(m["content"])
                    stats["user_chars"] += len(m["content"])

            record = {
                "messages": [{"role": "system", "content": cfg["system_prompt"]}]
                + [{"role": m["role"], "content": m["content"]} for m in compact],
                "meta": {
                    "dialog_id": f"{path.stem}:{compact[0].get('id')}",
                    "source_file": path.name,
                    "participant": name,
                    "time_start": compact[0]["time"].isoformat(),
                    "time_end": compact[-1]["time"].isoformat(),
                },
            }
            records.append(record)

    logging.info("Collected %d blocks before dedup", len(records))
    records = deduplicate(records, cfg, stats)
    logging.info(
        "After dedup: %d kept (exact removed=%d, near removed=%d)",
        len(records),
        stats["dedup_exact_removed"],
        stats["dedup_near_removed"],
    )

    final_metrics = compute_final_metrics(records)
    final_metrics["ratios"]["short_assistant_ratio"] = ratio(
        stats["short_assistant_total"], final_metrics["stats"]["assistant_messages"]
    )

    train, val = split_train_val(records, cfg)
    errors = validate_records(train) + validate_records(val)
    if errors:
        logging.warning("Validation errors: %d (first=%s)", len(errors), errors[0])

    output_dir = Path(cfg["output_dir"])
    write_jsonl(output_dir / cfg["train_file"], train)
    write_jsonl(output_dir / cfg["val_file"], val)
    write_jsonl(output_dir / cfg["excluded_file"], excluded_rows)

    sample_n = int(cfg.get("dry_run") or 0)
    if sample_n > 0:
        sample_n = min(sample_n, len(records))
        sample = random.sample(records, sample_n) if sample_n else []
        write_jsonl(output_dir / cfg["samples_file"], sample)

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": cfg,
        "stats": stats,
        "final_stats": final_metrics["stats"],
        "length_stats": final_metrics["length_stats"],
        "ratios": final_metrics["ratios"],
        "top_duplicates": final_metrics["top_duplicates"],
        "counts": {
            "records_total": len(records),
            "train_records": len(train),
            "val_records": len(val),
        },
        "validation_errors": errors[:50],
        "examples_excluded": stats["examples_excluded"][:20],
        "top_duplicate_pairs": stats["duplicates"][:20],
    }
    with (output_dir / cfg["report_file"]).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram SFT dataset builder")
    parser.add_argument("--config", help="Path to JSON config", default=None)
    parser.add_argument("--dry_run", type=int, help="Write N samples to samples.jsonl", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run is not None:
        cfg["dry_run"] = args.dry_run

    random.seed(int(cfg["split"].get("seed", 42)))
    setup_logging(Path(cfg["output_dir"]), cfg["log_file"])

    report = build_dataset(cfg)
    logging.info(
        "Done: train=%s val=%s report=%s",
        Path(cfg["output_dir"]) / cfg["train_file"],
        Path(cfg["output_dir"]) / cfg["val_file"],
        Path(cfg["output_dir"]) / cfg["report_file"],
    )
    logging.info("Stats: %s", report["counts"])


if __name__ == "__main__":
    main()
