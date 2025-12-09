import json
import logging
import random
import os
import glob
from datetime import datetime, timedelta
from collections import Counter

# ========== КОНСТАНТЫ НАСТРОЙКИ ==========

# Входные файлы: несколько result*.json в папке data/input
INPUT_DIR       = "data/input"
INPUT_GLOB      = "*.json"   # можно поменять на "*.json", если нужно

OUTPUT_JSONL        = "data/output/train.jsonl"             # основной SFT-dataset (train)
VALIDATION_JSONL    = "data/output/valid.jsonl"             # валидационная выборка
ORPHANS_JSONL       = "data/output/orphans.jsonl"           # блоки без корректной пары user–assistant
SHORT_ANS_JSONL     = "data/output/short_assistants.jsonl"  # блоки с слишком коротким ответом ассистента

TIME_WINDOW_HOURS   = 2        # окно контекста (часов) для одного диалога
MERGE_INTERVAL_MIN  = 5        # если подряд сообщения одного автора <= N минут — сливать
SYSTEM_PROMPT       = "Тебя зовут Влад. Ты человек, эксперт в сфере IT. Пиши короткие ответы, мотивируй продолжать диалог."
ASSISTANT_NAMES     = {"Vs"}   # имена/ники, считающиеся "assistant"
MIN_ASSISTANT_CHARS = 5        # минимальная длина последнего ответа ассистента
KEEP_SHORT_ASSISTANTS = True   # True — короткие ответы идут в train/valid, False — в отдельный файл
MIN_MESSAGES_IN_BLOCK = 2      # минимум сообщений в диалоге (после обработки)
MERGE_ASSISTANT_ALWAYS = True  # сливать подряд идущие assistant-сообщения

LOG_LEVEL = logging.INFO
LOG_FILE  = "data/output/build_dataset.log"

# ---- ПАРАМЕТРЫ ВАЛИДАЦИОННОГО СПЛИТА ----
VALIDATION_RATIO = 0.15         # доля блоков, попадающих в валидацию (0.0–1.0)
RANDOM_SEED      = 42          # фиксируем сид для воспроизводимости сплита

# ---- ПАРАМЕТРЫ ИНСТРУКЦИИ К USER-СООБЩЕНИЮ ----
USER_INSTRUCTION_TEMPLATE = "Имя пользователя: {who}. Напиши ответ на сообщение: {text}"

# ========================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=LOG_FILE,
    filemode="w",
    encoding="utf-8"
)

random.seed(RANDOM_SEED)


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
            # Если роль assistant и MERGE_ASSISTANT_ALWAYS — всегда сливаем
            if role == "assistant" and MERGE_ASSISTANT_ALWAYS:
                prev["content"] += "\n" + content
                meta["assistant_merges"] += 1
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


def apply_user_instruction(messages: list, user_name: str):
    """
    Трансформирует все user-сообщения в формате:
    'Имя пользователя: WHO. Напиши ответ на сообщение: TEXT'
    """
    for m in messages:
        if m["role"] == "user":
            original_text = m["content"]
            m["content"] = USER_INSTRUCTION_TEMPLATE.format(
                who=user_name,
                text=original_text
            )


def _get_person_stats(stats: dict, person: str) -> dict:
    """Возвращает (и при необходимости создаёт) статистику по конкретному собеседнику."""
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
    """
    dialogs_all: список словарей вида {"person": <имя/чат>, "block": <список сообщений>}
    """
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
        "per_person": {}
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
                # классифицируем причину
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

            # есть корректный блок
            # длина последнего ответа ассистента
            last_ass = [m for m in compact if m["role"] == "assistant"][-1]
            last_len = len(last_ass["content"])
            capped_len = min(last_len, 2000)
            stats["msg_len_counter"][capped_len] += 1
            pstats["msg_len_counter"][capped_len] += 1

            # копия compact для модификации user-сообщений
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
                # Сплит на train / validation
                target_file = out_main
                stats["kept_blocks"] += 1
                pstats["kept_blocks"] += 1

                if VALIDATION_RATIO > 0.0:
                    if random.random() < VALIDATION_RATIO:
                        target_file = out_valid
                        stats["val_blocks"] += 1
                        pstats["val_blocks"] += 1
                    else:
                        stats["train_blocks"] += 1
                        pstats["train_blocks"] += 1
                else:
                    stats["train_blocks"] += 1
                    pstats["train_blocks"] += 1

                write_jsonl_line(target_file, msg_obj)

    # ===== ГЛОБАЛЬНАЯ АНАЛИТИКА =====
    logging.info("====== ГЛОБАЛЬНАЯ АНАЛИТИКА ======")
    logging.info("Всего блоков (окон диалога): %d", stats["total_blocks"])
    logging.info("Всего корректных блоков (train+val): %d", stats["kept_blocks"])
    logging.info("  В train.jsonl: %d", stats["train_blocks"])
    logging.info("  В valid.jsonl: %d", stats["val_blocks"])
    logging.info("Всего в orphans.jsonl: %d", stats["orphans_total"])
    logging.info("В short_assistants.jsonl (<%d симв.): %d",
                 MIN_ASSISTANT_CHARS, stats["short_assistant_blocks"])

    logging.info("Причины отбраковки блоков:")
    logging.info("  Нет ответа ассистента (нет пары user→assistant): %d", stats["rule_no_assistant"])
    logging.info("  Слишком мало сообщений в блоке: %d", stats["rule_too_few_messages"])
    logging.info("  Только одна роль в блоке (только user или только assistant): %d", stats["rule_only_one_role"])
    logging.info("  Пустой блок после обработки: %d", stats["rule_empty_block"])
    logging.info("  Прочие причины: %d", stats["rule_other"])

    logging.info("Блоков с обрезанным user-хвостом (tail_trimmed): %d", stats["tail_trimmed_blocks"])
    logging.info("Количество слияний assistant-сообщений: %d", stats["assistant_merges"])

    if stats["msg_len_counter"]:
        total_msgs = sum(stats["msg_len_counter"].values())
        avg_len = sum(k * v for k, v in stats["msg_len_counter"].items()) / total_msgs
        logging.info("Средняя длина последнего ответа ассистента (глобально): %.1f символов", avg_len)

    # ===== ПЕРСОНАЛЬНАЯ АНАЛИТИКА =====
    logging.info("====== ПЕРСОНАЛЬНАЯ АНАЛИТИКА ПО СОБЕСЕДНИКАМ ======")
    for person, ps in stats["per_person"].items():
        logging.info("--- %s ---", person)
        logging.info(
            "  Блоков: всего=%d, корректных=%d (train=%d, val=%d), orphans=%d, short=%d",
            ps["total_blocks"],
            ps["kept_blocks"],
            ps["train_blocks"],
            ps["val_blocks"],
            ps["orphans_total"],
            ps["short_assistant_blocks"],
        )
        logging.info(
            "  Отбраковка: no_assistant=%d, too_few=%d, one_role=%d, empty=%d, other=%d",
            ps["rule_no_assistant"],
            ps["rule_too_few_messages"],
            ps["rule_only_one_role"],
            ps["rule_empty_block"],
            ps["rule_other"],
        )
        if ps["msg_len_counter"]:
            total_p = sum(ps["msg_len_counter"].values())
            avg_p = sum(k * v for k, v in ps["msg_len_counter"].items()) / total_p
            logging.info(
                "  Средняя длина последнего ответа ассистента: %.1f символов",
                avg_p
            )


def main():
    dialogs_all = []
    total_msgs = 0

    pattern = os.path.join(INPUT_DIR, INPUT_GLOB)
    files = sorted(glob.glob(pattern))
    logging.info("Найдено файлов с переписками: %d (шаблон: %s)", len(files), pattern)

    if not files:
        logging.warning("Не найдено ни одного файла по шаблону %s", pattern)

    for path in files:
        data = load_data(path)
        name = data.get("name") or os.path.basename(path)
        msgs = data.get("messages", [])
        total_msgs += len(msgs)
        dialogs = process_chat(msgs)

        for block in dialogs:
            dialogs_all.append({"person": name, "block": block})

        logging.info(
            "Файл '%s' (чат '%s'): %d сообщений → %d блоков",
            path, name, len(msgs), len(dialogs)
        )

    logging.info("Всего исходных сообщений (по всем файлам): %d", total_msgs)
    logging.info("Всего блоков после группировки по времени: %d", len(dialogs_all))

    generate_jsonl(dialogs_all)
    logging.info("Готово. Основной датасет (train): %s", OUTPUT_JSONL)
    logging.info("Валидационный датасет: %s", VALIDATION_JSONL)
    logging.info("Лог сохранён в файле: %s", LOG_FILE)


if __name__ == "__main__":
    main()
