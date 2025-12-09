import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DATA_PATH = Path("data/output/train.jsonl")
MAX_SEQ_LEN = 2048
OUT_FILE = Path("test.txt")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
eos_id = tokenizer.eos_token_id


def to_alternating(messages: List[Dict]) -> List[Dict]:
    merged = []
    for m in messages:
        content = m["content"].strip()
        if not content:
            continue
        role = m["role"]
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n" + content
        else:
            merged.append({"role": role, "content": content})

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


def format_chat(messages: List[Dict]) -> (str, str):
    ass_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not ass_indices:
        return "", ""
    last_ass_idx = ass_indices[-1]
    answer = messages[last_ass_idx]["content"].strip()
    if not answer:
        return "", ""
    chat_ctx = to_alternating(messages[:last_ass_idx])
    if not chat_ctx:
        return "", ""
    prompt = tokenizer.apply_chat_template(
        chat_ctx,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, answer


def build_samples(raw: List[Dict]):
    for rec in raw:
        msgs = rec["messages"]
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
            answer_ids = answer_ids[: MAX_SEQ_LEN - 1]
            max_prompt_len = 0

        if not prompt_ids:
            prompt_chunks = [[]]
        elif len(prompt_ids) <= max_prompt_len or max_prompt_len == 0:
            prompt_chunks = [prompt_ids[-max_prompt_len:]]
        else:
            stride = max(max_prompt_len // 2, 1)
            n = len(prompt_ids)
            prompt_chunks = []
            start = 0
            while start < n:
                chunk = prompt_ids[start : start + max_prompt_len]
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
            yield {
                "prompt_text": tokenizer.decode(chunk, skip_special_tokens=True),
                "answer_text": tokenizer.decode(answer_ids, skip_special_tokens=True),
                "input_len": len(input_ids),
            }


def main():
    raw = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8")]
    lines = []
    for i, sample in enumerate(build_samples(raw), 1):
        line = (
            f"#{i} len={sample['input_len']}\n"
            f"PROMPT:\n{sample['prompt_text']}\n"
            f"ANSWER:\n{sample['answer_text']}\n"
            f"{'-'*40}"
        )
       # print(line)
        lines.append(line)
        break
    OUT_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
