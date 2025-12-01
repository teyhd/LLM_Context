import json
import logging
from datetime import datetime, timedelta
from collections import Counter

# ========== КОНСТАНТЫ НАСТРОЙКИ ==========

INPUT_JSON          = "data/input/result.json"        # исходный экспорт Telegram
OUTPUT_JSONL        = "data/output/train.jsonl"        # основной SFT-дataset
ORPHANS_JSONL       = "data/output/orphans.jsonl"      # блоки без корректной пары user–assistant
SHORT_ANS_JSONL     = "data/output/short_assistants.jsonl"  # блоки с слишком коротким ответом ассистента

TIME_WINDOW_HOURS   = 2        # окно контекста (часов) для одного диалога
MERGE_INTERVAL_MIN  = 5        # если подряд сообщения одного автора <= N минут — сливать
SYSTEM_PROMPT       = "Ты — вежливый и полезный ассистент."
ASSISTANT_NAMES     = {"Vs"}   # имена/ники, считающиеся "assistant"
MIN_ASSISTANT_CHARS = 5        # минимальная длина последнего ответа ассистента
KEEP_SHORT_ASSISTANTS = False  # True — короткие ответы идут в train, False — в отдельный файл
MIN_MESSAGES_IN_BLOCK = 2      # минимум сообщений в диалоге (после обработки)
MERGE_ASSISTANT_ALWAYS = True  # сливать подряд идущие assistant-сообщения
                               # даже если интервал > MERGE_INTERVAL_MIN (но внутри блока)
LOG_LEVEL = logging.INFO

# ========================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_datetime(s: str):
    """Парсер дат Telegram (ISO-формат)."""
    if not s:
        return None
    try:
        # на всякий случай заменим Z на +00:00
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception as e:
        logging.warning("Не удалось распарсить дату '%s': %s", s, e)
        return None


def flatten_text(msg) -> str | None:
    """
    Telegram text может быть:
    - строкой
    - списком частей (строки и объекты с ключом "text")
    """
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
    """Определяем, чьё сообщение считаем ответом ассистента."""
    return msg.get("from") in ASSISTANT_NAMES


def process_chat(messages: list):
    """
    Группировка сообщений чата в блоки-диалоги:
    - сортировка по времени
    - разбиение на окна по TIME_WINDOW_HOURS
    - слияние подряд сообщений одного автора при интервале <= MERGE_INTERVAL_MIN
    """
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

        # новое окно контекста
        if (m["time"] - last_time) > timedelta(hours=TIME_WINDOW_HOURS):
            dialogs.append(current)
            current = [m]
            last_time = m["time"]
            continue

        # слияние подряд сообщений одного автора, если интервалы короткие
        if current[-1]["role"] == m["role"] and (m["time"] - last_time) <= timedelta(minutes=MERGE_INTERVAL_MIN):
            current[-1]["content"] += "\n" + m["content"]
        else:
            current.append(m)

        last_time = m["time"]

    if current:
        dialogs.append(current)
    return dialogs


def shape_block(block: list, stats: dict):
    """
    Постобработка диалога:
    - если нет ассистента → reject (no_assistant)
    - отрезаем хвост user-сообщений до последнего assistant
    - сливаем подряд идущие одинаковые роли (с учётом MERGE_ASSISTANT_ALWAYS)
    - отбрасываем блоки:
       * слишком короткие по кол-ву сообщений
       * с единственной ролью (нет user или нет assistant)
    Возвращает (compact_block | None, reason_rejected | None, meta_info)
    meta_info: { "tail_trimmed": bool, "assistant_merges": int }
    """
    meta = {"tail_trimmed": False, "assistant_merges": 0}

    if not block:
        return None, "empty_block", meta

    roles_original = {m["role"] for m in block}
    if "assistant" not in roles_original:
        return None, "no_assistant", meta

    # отрезаем хвост user-ов после последнего assistant
    last_ass_idx = max(i for i, m in enumerate(block) if m["role"] == "assistant")
    if last_ass_idx < len(block) - 1:
        meta["tail_trimmed"] = True
    block = block[:last_ass_idx + 1]

    # убираем time и сжимаем подряд идущие роли
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
        # Мержим, если предыдущий того же role
        if prev["role"] == role:
            # Если раль assistant и MERGE_ASSISTANT_ALWAYS — всегда сливаем
            if role == "assistant" and MERGE_ASSISTANT_ALWAYS:
                prev["content"] += "\n" + content
                meta["assistant_merges"] += 1
            # если не assistant — логика как раньше: слили уже на уровне process_chat
            else:
                prev["content"] += "\n" + content
        else:
            compact.append({"role": role, "content": content})

    # Фильтры по размеру и ролям
    if len(compact) < MIN_MESSAGES_IN_BLOCK:
        return None, "too_few_messages", meta

    roles_compact = {m["role"] for m in compact}
    if not ("user" in roles_compact and "assistant" in roles_compact):
        return None, "only_one_role", meta

    return compact, None, meta


def write_jsonl_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_jsonl(dialogs_all: list):
    stats = {
        "total_blocks": 0,
        "kept_blocks": 0,
        "orphans_total": 0,
        "short_assistant_blocks": 0,
        "rule_no_assistant": 0,
        "rule_too_few_messages": 0,
        "rule_only_one_role": 0,
        "rule_empty_block": 0,
        "rule_other": 0,
        "tail_trimmed_blocks": 0,
        "assistant_merges": 0,
        "msg_len_counter": Counter()
    }

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_main, \
         open(ORPHANS_JSONL, "w", encoding="utf-8") as out_orphans, \
         open(SHORT_ANS_JSONL, "w", encoding="utf-8") as out_short:

        for block in dialogs_all:
            stats["total_blocks"] += 1

            compact, reason, meta = shape_block(block, stats)
            if meta["tail_trimmed"]:
                stats["tail_trimmed_blocks"] += 1
            stats["assistant_merges"] += meta["assistant_merges"]

            if compact is None:
                # классифицируем причину
                if reason == "no_assistant":
                    stats["rule_no_assistant"] += 1
                elif reason == "too_few_messages":
                    stats["rule_too_few_messages"] += 1
                elif reason == "only_one_role":
                    stats["rule_only_one_role"] += 1
                elif reason == "empty_block":
                    stats["rule_empty_block"] += 1
                else:
                    stats["rule_other"] += 1

                msg_obj = {
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT}] +
                                [{"role": m["role"], "content": m["content"]} for m in block]
                }
                write_jsonl_line(out_orphans, msg_obj)
                stats["orphans_total"] += 1
                continue

            # есть корректный блок
            # длина последнего ответа ассистента
            last_ass = [m for m in compact if m["role"] == "assistant"][-1]
            last_len = len(last_ass["content"])
            stats["msg_len_counter"][min(last_len, 2000)] += 1

            msg_obj = {
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + compact
            }

            if not KEEP_SHORT_ASSISTANTS and last_len < MIN_ASSISTANT_CHARS:
                write_jsonl_line(out_short, msg_obj)
                stats["short_assistant_blocks"] += 1
            else:
                write_jsonl_line(out_main, msg_obj)
                stats["kept_blocks"] += 1

    # ===== АНАЛИТИКА =====
    logging.info("====== АНАЛИТИКА ======")
    logging.info("Всего блоков (окон диалога): %d", stats["total_blocks"])
    logging.info("В train.jsonl (kept_blocks): %d", stats["kept_blocks"])
    logging.info("Всего в orphans.jsonl: %d", stats["orphans_total"])
    logging.info("В short_assistants.jsonl (<%d симв.): %d",
                 MIN_ASSISTANT_CHARS, stats["short_assistant_blocks"])

    logging.info("Причины отбраковки блоков:")
    logging.info("  no_assistant       : %d", stats["rule_no_assistant"])
    logging.info("  too_few_messages   : %d", stats["rule_too_few_messages"])
    logging.info("  only_one_role      : %d", stats["rule_only_one_role"])
    logging.info("  empty_block        : %d", stats["rule_empty_block"])
    logging.info("  other              : %d", stats["rule_other"])

    logging.info("Блоков с обрезанным user-хвостом (tail_trimmed): %d", stats["tail_trimmed_blocks"])
    logging.info("Количество слияний assistant-сообщений: %d", stats["assistant_merges"])

    if stats["msg_len_counter"]:
        total_msgs = sum(stats["msg_len_counter"].values())
        avg_len = sum(k * v for k, v in stats["msg_len_counter"].items()) / total_msgs
        logging.info("Средняя длина последнего ответа ассистента: %.1f символов", avg_len)


def main():
    data = load_data(INPUT_JSON)
    dialogs_all = []
    total_msgs = 0

    chats = data.get("chats", {}).get("list", [])
    logging.info("Найдено чатов: %d", len(chats))

    name = data.get("name", "UNKNOWN")
    msgs = data.get("messages", [])
    total_msgs += len(msgs)
    dialogs = process_chat(msgs)
    dialogs_all.extend(dialogs)
    logging.info("Чат '%s': %d сообщений → %d блоков", name, len(msgs), len(dialogs))

    logging.info("Всего исходных сообщений: %d", total_msgs)
    logging.info("Всего блоков после группировки по времени: %d", len(dialogs_all))

    generate_jsonl(dialogs_all)
    logging.info("Готово. Основной датасет: %s", OUTPUT_JSONL)


if __name__ == "__main__":
    main()