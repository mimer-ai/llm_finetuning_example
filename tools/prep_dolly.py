# tools/prep_dolly.py
"""
Download and convert databricks/databricks-dolly-15k to our JSONL format.
Creates data/train.jsonl and data/val.jsonl. Supports subset selection.
"""

import argparse
import json
import os
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap on training examples",
    )
    ap.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Optional cap on validation examples",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    ds = ds.shuffle(seed=args.seed)
    n = len(ds)
    cut = int(args.train_frac * n)

    train = ds.select(range(cut))
    val = ds.select(range(cut, n))

    if args.max_train_samples:
        train = train.select(range(min(args.max_train_samples, len(train))))
    if args.max_val_samples:
        val = val.select(range(min(args.max_val_samples, len(val))))

    out_train = os.path.join(args.out_dir, "train.jsonl")
    out_val = os.path.join(args.out_dir, "val.jsonl")

    with open(out_train, "w", encoding="utf-8") as ft:
        for r in train:
            rec = {
                "instruction": r["instruction"],
                "input": r.get("context", "") or "",
                "output": r["response"],
            }
            ft.write(json.dumps(rec, ensure_ascii=False) + "")

    with open(out_val, "w", encoding="utf-8") as fv:
        for r in val:
            rec = {
                "instruction": r["instruction"],
                "input": r.get("context", "") or "",
                "output": r["response"],
            }
            fv.write(json.dumps(rec, ensure_ascii=False) + "")

    print("Wrote", out_train, out_val)


if __name__ == "__main__":
    main()
