import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a LoRA adapter")
    parser.add_argument("--base", required=True, help="Base model id or path")
    parser.add_argument("--adapter", required=True, help="Adapter directory")
    parser.add_argument("--load_4bit", action="store_true", help="Load base model in 4-bit")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    quant_cfg = None
    if args.load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_cfg,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print("Ready. Type a prompt, Ctrl+C to exit.")
    while True:
        try:
            prompt = input("\nUSER> ").strip()
            if not prompt:
                continue
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0][enc.input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"ASSISTANT> {text}")
        except KeyboardInterrupt:
            print("\nBye.")
            return


if __name__ == "__main__":
    main()
