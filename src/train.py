# src/train.py
import math
import argparse
import random
import os
import sys
from dataclasses import dataclass
from typing import Any
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_prompt(instruction: str, inp: str) -> str:
    instruction = instruction or ""
    inp = inp or ""

    if isinstance(instruction, str) is False:
        instruction = ""
    if isinstance(inp, str) is False:
        inp = ""

    instruction = instruction.strip()
    inp = inp.strip()

    if inp != "":
        return (
            "### Instruction:\n"
            + instruction
            + "\n\n### Input:\n"
            + inp
            + "\n\n### Response:\n"
        )
    else:
        return "### Instruction:\n" + instruction + "\n\n### Response:\n"


@dataclass
class DataCollatorCausalLMWithLabels:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_pad = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch_pad["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = torch.tensor(f["labels"], dtype=torch.long)
            pad_len = max_len - lab.shape[0]
            if pad_len > 0:
                lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            labels.append(lab)
        return {
            "input_ids": batch_pad["input_ids"],
            "attention_mask": batch_pad["attention_mask"],
            "labels": torch.stack(labels),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train_file", type=str, default="data/train.jsonl")
    p.add_argument("--val_file", type=str, default="data/val.jsonl")
    p.add_argument("--output_dir", type=str, default="outputs/qwen15b_lora_dolly")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--per_device_batch_size", type=int, default=6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16" if args.bf16 else "fp16")
    if not accelerator.is_main_process:
        import transformers

        transformers.utils.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        devnull = getattr(os, "devnull", "/dev/null")
        sys.stdout = open(devnull, "w")
        sys.stderr = open(devnull, "w")

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc_in", "fc_out"],
    )
    model = get_peft_model(model, lora)

    dataset = load_dataset(
        "json", data_files={"train": args.train_file, "validation": args.val_file}
    )
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    def preprocess(example):
        def safe_str(x):
            if isinstance(x, str) and x.strip() != "":
                return x
            return " "

        instr = safe_str(example.get("instruction"))
        inp = safe_str(example.get("input"))
        out = safe_str(example.get("output"))
        prompt = build_prompt(instr, inp)

        # Build messages for Qwen chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": out},
        ]

        # Tokenize fully (prompt + response)
        tokenized_full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

        input_ids = tokenized_full

        # Tokenize system+user only to find the boundary
        messages_no_assistant = messages[:-1]
        tokenized_prompt_only = tokenizer.apply_chat_template(
            messages_no_assistant, tokenize=True, add_generation_prompt=True
        )

        boundary = len(tokenized_prompt_only)

        # Labels
        labels = [-100] * len(input_ids)
        labels[boundary:] = input_ids[boundary:]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    tokenized = dataset.map(preprocess, remove_columns=dataset["train"].column_names)
    train_ds = tokenized["train"]
    val_ds = tokenized["validation"]
    collator = DataCollatorCausalLMWithLabels(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(param_groups, lr=args.learning_rate)
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.is_main_process and step % 50 == 0:
                print(
                    f"Epoch {epoch} Step {step}/{len(train_loader)} | loss {loss.item():.4f}"
                )
        model.eval()
        eval_loss = 0.0
        nb = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(**batch)
                eval_loss += outputs.loss.item() * batch["input_ids"].shape[0]
                nb += batch["input_ids"].shape[0]
        eval_loss /= max(1, nb)
        ppl = math.exp(eval_loss) if eval_loss < 30 else float("inf")
        if accelerator.is_main_process:
            print(f"[Epoch {epoch}] eval_loss={eval_loss:.4f} perplexity={ppl:.2f}")
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
    if accelerator.is_main_process:
        print("Training finished. LoRA adapter saved to:", args.output_dir)


if __name__ == "__main__":
    main()
