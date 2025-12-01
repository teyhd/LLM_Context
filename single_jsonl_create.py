import json
import csv
import os
import logging
from datetime import datetime, timedelta

# ========== НАСТРОЙКИ (константы) ==========
INPUT_JSON  = "result.json"      # исходный экспорт Telegram
OUTPUT_JSONL = "train.jsonl"     # основной датасет
ORPHANS_JSONL = "orphans.jsonl"  # сообщения без пары ответов
TIME_WINDOW_HOURS = 2            # контекст: сколько часов переписка в одном блоке
MERGE_INTERVAL_MIN = 5           # если сообщения одного пользователя идут подряд с интервалом ≤ 5 мин — объединять

SYSTEM_PROMPT = "Ты — вежливый и полезный ассистент."
ASSISTANT_NAMES = {"Vs", "Ваш_ник", "MyName", "..."}  
# — множество имён, которые считаются «я», сообщения которых будут роль assistant.

# =========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_text(msg):
    # Если text — строка, возвращаем. Если список частей — конкатенируем. Иначе — None.
    t = msg.get("text")
    if isinstance(t, str):
        return t
    if isinstance(t, list):
        parts = []
        for p in t:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return "".join(parts)
    return None

def process_chat(messages):
    # Сортируем по дате
    messages = sorted(messages, key=lambda m: m.get("date"))
    dialogs = []
    current = []
    last_time = None

    for msg in messages:
        ts = msg.get("date")
        try:
            dt = datetime.fromisoformat(ts)
        except Exception as e:
            logging.warning("Не удалось распарсить дату %s: %s", ts, e)
            continue
        text = flatten_text(msg)
        if not text:
            continue

        role = "assistant" if msg.get("from") in ASSISTANT_NAMES else "user"

        # если текущий блок пуст — старт нового
        if not current:
            current = [{"role": role, "content": text, "time": dt}]
            last_time = dt
            continue

        # если интервал слишком большой — завершаем блок, сохраняем, начинаем новый
        if (dt - last_time) > timedelta(hours=TIME_WINDOW_HOURS):
            dialogs.append(current)
            current = [{"role": role, "content": text, "time": dt}]
            last_time = dt
            continue

        # если тот же пользователь, и интервал ≤ MERGE_INTERVAL_MIN — merge
        if current[-1]["role"] == role and (dt - last_time) <= timedelta(minutes=MERGE_INTERVAL_MIN):
            current[-1]["content"] += "\n" + text
            last_time = dt
            continue

        # иначе просто добавляем новое сообщение
        current.append({"role": role, "content": text, "time": dt})
        last_time = dt

    if current:
        dialogs.append(current)
    return dialogs

def generate_jsonl(dialogs, output_file, orphans_file=None):
    with open(output_file, "w", encoding="utf-8") as out_main, \
         open(orphans_file, "w", encoding="utf-8") if orphans_file else nullcontext() as out_orphans:
        for block in dialogs:
            # находим последние сообщения — если есть assistant — формируем записку
            has_assistant = any(m["role"] == "assistant" for m in block)
            if not has_assistant:
                if orphans_file:
                    out_orphans.write(json.dumps({"messages":[
                        {"role":"system","content":SYSTEM_PROMPT}
                    ] + [{"role":m["role"], "content":m["content"]} for m in block]}, ensure_ascii=False) + "\n")
                continue

            # строим объект messages
            msgs = [{"role":"system","content":SYSTEM_PROMPT}]
            for m in block:
                msgs.append({"role": m["role"], "content": m["content"]})
            out_main.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")

def main():
    data = load_data(INPUT_JSON)
    total = 0
    dialogs_all = []
    print(f"Processing chat: {data.get('name')}")
    msgs = data.get("messages", [])
    dialogs = process_chat(msgs)
    dialogs_all.extend(dialogs)
    logging.info("Чат %s: %d сообщений → %d диалогов", data.get("name"), len(msgs), len(dialogs))
    total += len(msgs)
    logging.info("Всего сообщений: %d, диалогов: %d", total, len(dialogs_all))

    generate_jsonl(dialogs_all, OUTPUT_JSONL, ORPHANS_JSONL)
    logging.info("Готово. Записаны файлы %s и %s", OUTPUT_JSONL, ORPHANS_JSONL)

if __name__ == "__main__":
    main()