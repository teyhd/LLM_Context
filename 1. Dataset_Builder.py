import json
import logging
import random
import os
import glob
from datetime import datetime, timedelta
from collections import Counter

# ========== КОНСТАНТЫ НАСТРОЙКИ ==========

INPUT_DIR       = "data/input"
INPUT_GLOB      = "*.json"

OUTPUT_JSONL        = "data/output/train.jsonl"
VALIDATION_JSONL    = "data/output/valid.jsonl"
ORPHANS_JSONL       = "data/output/orphans.jsonl"
SHORT_ANS_JSONL     = "data/output/short_assistants.jsonl"

TIME_WINDOW_HOURS   = 1
MERGE_INTERVAL_MIN  = 5

SYSTEM_PROMPT    = "Ты Влад. Ты дружелюбный и лаконичный.\nГлавный фокус — переписка: отвечай по делу, без лишней воды."
ASSISTANT_NAMES     = {"Vs"}
MIN_ASSISTANT_CHARS = 5
KEEP_SHORT_ASSISTANTS = True
MIN_MESSAGES_IN_BLOCK = 2
MERGE_ASSISTANT_ALWAYS = True

LOG_LEVEL = logging.INFO
LOG_FILE  = "data/output/build_dataset.log"

VALIDATION_RATIO = 0.15
RANDOM_SEED      = 42

USER_INSTRUCTION_TEMPLATE = "Имя собеседника: {who}. Напиши ответ на сообщение: {text}"

# ========================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=LOG_FILE,
    filemode="w",
    encoding="utf-8",
)

random.seed(RANDOM_SEED)


def load_data(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="cp1251") as f:
            return json.load(f)


def parse_datetime(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception as e:  # noqa: BLE001
        logging.warning("Не удалось распарсить дату '%s': %s", s, e)
        return None


def flatten_text(msg) -> str | None:
    t = msg.get("text")
    if isinstance(t, str):
        text = t.strip()
        return text or None
    if isinstance(t, list):
        parts = []
        for p in t:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
        text = "".join(parts).strip()
        return text or None
    return None


def is_assistant(msg) -> bool:
    return msg.get("from") in ASSISTANT_NAMES


def process_chat(messages: list):
    prepared = []
    for m in messages:
        dt = parse_datetime(m.get("date"))
        if not dt:
            continue
        text = flatten_text(m)
        if not text:
            continue
        role = "assistant" if is_assistant(m) else "user"
        prepared.append({"role": role, "content": text, "time": dt})

    if not prepared:
        return []

    prepared.sort(key=lambda x: x["time"])

    dialogs = []
    current = []
    last_time = None

    for m in prepared:
        if not current:
            current = [m]
            last_time = m["time"]
            continue

        if (m["time"] - last_time) > timedelta(hours=TIME_WINDOW_HOURS):
            dialogs.append(current)
            current = [m]
            last_time = m["time"]
            continue

        if current[-1]["role"] == m["role"] and (m["time"] - last_time) <= timedelta(minutes=MERGE_INTERVAL_MIN):
            current[-1]["content"] += "\n" + m["content"]
        else:
            current.append(m)

        last_time = m["time"]

    if current:
        dialogs.append(current)
    return dialogs


def shape_block(block: list, stats: dict):
    meta = {"tail_trimmed": False, "assistant_merges": 0}

    if not block:
        return None, "empty_block", meta

    roles_original = {m["role"] for m in block}
    if "assistant" not in roles_original:
        return None, "no_assistant", meta

    last_ass_idx = max(i for i, m in enumerate(block) if m["role"] == "assistant")
    if last_ass_idx < len(block) - 1:
        meta["tail_trimmed"] = True
    block = block[: last_ass_idx + 1]

    compact = []
    for m in block:
        role = m["role"]
        content = m["content"].strip()
        if not content:
            continue

        if not compact:
            compact.append({"role": role, "content": content})
            continue

        prev = compact[-1]
        if prev["role"] == role:
            prev["content"] += "\n" + content
            if role == "assistant":
                meta["assistant_merges"] += 1
        else:
            compact.append({"role": role, "content": content})

    if len(compact) < MIN_MESSAGES_IN_BLOCK:
        return None, "too_few_messages", meta

    roles_compact = {m["role"] for m in compact}
    if not ("user" in roles_compact and "assistant" in roles_compact):
        return None, "only_one_role", meta

    return compact, None, meta


def write_jsonl_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def apply_user_instruction(messages: list, user_name: str):
    for m in messages:
        if m["role"] == "user":
            original_text = m["content"]
            m["content"] = USER_INSTRUCTION_TEMPLATE.format(who=user_name, text=original_text)


def _get_person_stats(stats: dict, person: str) -> dict:
    per = stats["per_person"]
    if person not in per:
        per[person] = {
            "total_blocks": 0,
            "kept_blocks": 0,
            "train_blocks": 0,
            "val_blocks": 0,
            "orphans_total": 0,
            "short_assistant_blocks": 0,
            "rule_no_assistant": 0,
            "rule_too_few_messages": 0,
            "rule_only_one_role": 0,
            "rule_empty_block": 0,
            "rule_other": 0,
            "msg_len_counter": Counter(),
        }
    return per[person]


def generate_jsonl(dialogs_all: list):
    stats = {
        "total_blocks": 0,
        "kept_blocks": 0,
        "train_blocks": 0,
        "val_blocks": 0,
        "orphans_total": 0,
        "short_assistant_blocks": 0,
        "rule_no_assistant": 0,
        "rule_too_few_messages": 0,
        "rule_only_one_role": 0,
        "rule_empty_block": 0,
        "rule_other": 0,
        "tail_trimmed_blocks": 0,
        "assistant_merges": 0,
        "msg_len_counter": Counter(),
        "per_person": {},
    }

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_main, \
         open(VALIDATION_JSONL, "w", encoding="utf-8") as out_valid, \
         open(ORPHANS_JSONL, "w", encoding="utf-8") as out_orphans, \
         open(SHORT_ANS_JSONL, "w", encoding="utf-8") as out_short:

        for entry in dialogs_all:
            person = entry.get("person", "UNKNOWN")
            block = entry["block"]

            stats["total_blocks"] += 1
            pstats = _get_person_stats(stats, person)
            pstats["total_blocks"] += 1

            compact, reason, meta = shape_block(block, stats)
            if meta["tail_trimmed"]:
                stats["tail_trimmed_blocks"] += 1
            stats["assistant_merges"] += meta["assistant_merges"]

            if compact is None:
                if reason == "no_assistant":
                    stats["rule_no_assistant"] += 1
                    pstats["rule_no_assistant"] += 1
                elif reason == "too_few_messages":
                    stats["rule_too_few_messages"] += 1
                    pstats["rule_too_few_messages"] += 1
                elif reason == "only_one_role":
                    stats["rule_only_one_role"] += 1
                    pstats["rule_only_one_role"] += 1
                elif reason == "empty_block":
                    stats["rule_empty_block"] += 1
                    pstats["rule_empty_block"] += 1
                else:
                    stats["rule_other"] += 1
                    pstats["rule_other"] += 1

                msg_obj = {
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] +
                                [{"role": m["role"], "content": m["content"]} for m in block]
                }
                write_jsonl_line(out_orphans, msg_obj)
                stats["orphans_total"] += 1
                pstats["orphans_total"] += 1
                continue

            last_ass = [m for m in compact if m["role"] == "assistant"][-1]
            last_len = len(last_ass["content"])
            capped_len = min(last_len, 2000)
            stats["msg_len_counter"][capped_len] += 1
            pstats["msg_len_counter"][capped_len] += 1

            msgs_for_json = [dict(m) for m in compact]
            apply_user_instruction(msgs_for_json, person)

            msg_obj = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs_for_json
            }

            if not KEEP_SHORT_ASSISTANTS and last_len < MIN_ASSISTANT_CHARS:
                write_jsonl_line(out_short, msg_obj)
                stats["short_assistant_blocks"] += 1
                pstats["short_assistant_blocks"] += 1
            else:
                target_file = out_main
                stats["kept_blocks"] += 1
                pstats["kept_blocks"] += 1

                if VALIDATION_RATIO > 0.0 and random.random() < VALIDATION_RATIO:
                    target_file = out_valid
                    stats["val_blocks"] += 1
                    pstats["val_blocks"] += 1
                else:
                    stats["train_blocks"] += 1
                    pstats["train_blocks"] += 1

                write_jsonl_line(target_file, msg_obj)

    logging.info("====== Итоги ======")
    logging.info("Блоков всего: %d", stats["total_blocks"])
    logging.info("Принято (train+val): %d", stats["kept_blocks"])
    logging.info("  train.jsonl: %d", stats["train_blocks"])
    logging.info("  valid.jsonl: %d", stats["val_blocks"])
    logging.info("В orphans.jsonl: %d", stats["orphans_total"])
    logging.info("В short_assistants.jsonl (<%d): %d", MIN_ASSISTANT_CHARS, stats["short_assistant_blocks"])
    logging.info("tail_trimmed: %d | assistant_merges: %d", stats["tail_trimmed_blocks"], stats["assistant_merges"])

    # Развёрнутая аналитика по собеседникам (сортировка по принятому количеству)
    persons_sorted = sorted(
        stats["per_person"].items(),
        key=lambda kv: kv[1]["kept_blocks"],
        reverse=True,
    )
    for person, ps in persons_sorted:
        orphan_rate = (ps["orphans_total"] / ps["total_blocks"] * 100) if ps["total_blocks"] else 0.0
        kept_rate = (ps["kept_blocks"] / ps["total_blocks"] * 100) if ps["total_blocks"] else 0.0
        logging.info(
            "[%s] всего=%d | принят=%d (train=%d, val=%d, %.1f%%) | orphans=%d (%.1f%%) | short=%d",
            person,
            ps["total_blocks"],
            ps["kept_blocks"],
            ps["train_blocks"],
            ps["val_blocks"],
            kept_rate,
            ps["orphans_total"],
            orphan_rate,
            ps["short_assistant_blocks"],
        )


def main():
    dialogs_all = []
    total_msgs = 0

    pattern = os.path.join(INPUT_DIR, INPUT_GLOB)
    files = sorted(glob.glob(pattern))
    logging.info("Найдено файлов: %d (маска %s)", len(files), pattern)
    if not files:
        logging.warning("Нет входных файлов под маску %s", pattern)

    for path in files:
        data = load_data(path)
        name = data.get("name") or os.path.basename(path)
        msgs = data.get("messages", [])
        total_msgs += len(msgs)
        dialogs = process_chat(msgs)
        for block in dialogs:
            dialogs_all.append({"person": name, "block": block})
        logging.info("Обработан %s (%s): %d сообщений -> %d блоков", path, name, len(msgs), len(dialogs))

    logging.info("Сообщений всего: %d", total_msgs)
    logging.info("Блоков всего: %d", len(dialogs_all))

    generate_jsonl(dialogs_all)
    logging.info("Готово: train=%s, valid=%s, log=%s", OUTPUT_JSONL, VALIDATION_JSONL, LOG_FILE)


if __name__ == "__main__":
    main()
