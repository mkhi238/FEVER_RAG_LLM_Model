"""Train one task's LoRA adapter on the shared deberta-v3 NLI base (tasks.py), from task_data.py's jsonl.

  python -m claimscan.verifier.train_task_adapter --task define --max-steps 200 --limit 2000   # smoke run
  python -m claimscan.verifier.train_task_adapter --task define --separate                    # full fine-tune instead

Default: LoRA (r=8) on the attention projections (query_proj, key_proj, value_proj); the 3-way NLI classifier is
trained too (PEFT's modules_to_save for SEQ_CLS), starting from the base's own NLI head, so labels keep the base's
entailment / neutral / contradiction columns. Saves to <artifacts>/<natlog.adapter_dir>/<task>/ (adapter weights +
train_log.json with the loss curve). Point natlog.tasks.<task>.adapter at that dir to use it. --separate saves a full
model there instead (natlog.tasks.<task>.model). Reports the training loss only; dev scoring is a proposed eval that
waits for the user's OK (docs/NATLOG_FOL.md).
Memory: base fp32 ~0.7 GB; batch 8 x 256 tokens with fp16 autocast fits the RTX 2060. Check `free -g` first.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

LORA_TARGETS = ["query_proj", "key_proj", "value_proj"]


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
  rows = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      rows.append(json.loads(line))
      if limit and len(rows) >= limit:
        break
  return rows


def label_ids(model_config) -> dict[str, int]:
  """'entailment' / 'contradiction' / 'neutral' -> the base model's output column."""
  ids = {l.lower(): int(i) for i, l in model_config.id2label.items()}
  missing = {"entailment", "contradiction", "neutral"} - set(ids)
  if missing:
    raise ValueError(f"base model has no NLI label(s) {sorted(missing)}: {model_config.id2label}")
  return ids


def wrap_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1, targets=LORA_TARGETS):
  from peft import LoraConfig, TaskType, get_peft_model
  cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
  return get_peft_model(model, cfg)


def train_loop(model, batches, max_steps: int, lr: float, device: str, log_every: int = 20, fp16: bool = True):
  """Plain AdamW loop over an iterator of dict batches (input_ids, attention_mask, labels). Returns the loss log."""
  import torch
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.AdamW(params, lr=lr)
  sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, max_steps // 10)))
  use_amp = fp16 and device.startswith("cuda")
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
  model.train().to(device)
  log, running = [], []
  for step, batch in enumerate(batches, 1):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
      loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    sched.step()
    running.append(loss.item())
    if step % log_every == 0 or step == max_steps:
      log.append({"step": step, "loss": sum(running) / len(running)})
      print(f"step {step}: loss {log[-1]['loss']:.4f}", flush=True)
      running = []
    if step >= max_steps:
      break
  return log


def batches_from(rows, tokenizer, lab: dict[str, int], batch_size: int, max_length: int, seed: int):
  """Endless shuffled batches."""
  import torch
  rng = random.Random(seed)
  while True:
    order = list(range(len(rows)))
    rng.shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
      chunk = [rows[j] for j in order[i:i + batch_size]]
      enc = tokenizer([r["premise"] for r in chunk], [r["hypothesis"] for r in chunk], truncation=True,
                      max_length=max_length, padding=True, return_tensors="pt")
      enc.pop("token_type_ids", None)
      enc["labels"] = torch.tensor([lab[r["label"]] for r in chunk])
      yield dict(enc)


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--task", required=True, choices=("define", "example", "explain", "question", "operators"))
  ap.add_argument("--max-steps", type=int, default=200)
  ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows")
  ap.add_argument("--batch-size", type=int, default=8)
  ap.add_argument("--max-length", type=int, default=256)
  ap.add_argument("--lr", type=float, default=2e-4, help="2e-4 for LoRA; use ~2e-5 with --separate")
  ap.add_argument("--r", type=int, default=8)
  ap.add_argument("--separate", action="store_true", help="full fine-tune into a separate model (no LoRA)")
  ap.add_argument("--device", default=None, help="default: cuda if available")
  ap.add_argument("--out", default=None, help="default: <artifacts>/<natlog.adapter_dir>/<task>")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)

  import torch
  from transformers import AutoModelForSequenceClassification, AutoTokenizer

  from claimscan.config import config as cfg
  from claimscan.neurosymbolic_layer.fol.verify import settings
  from claimscan.verifier.task_data import task_data_dir

  s = settings(cfg)
  base = s.get("base_model") or cfg.verifier_inference_model
  data = task_data_dir(cfg) / args.task / "train.jsonl"
  if not data.exists():
    raise SystemExit(f"{data} missing: run python -m claimscan.verifier.task_data --task {args.task}")
  out = Path(args.out) if args.out else Path(cfg.artifacts_root) / s.get("adapter_dir", "natlog/adapters") / args.task
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(args.seed)

  rows = read_jsonl(data, args.limit)
  tok = AutoTokenizer.from_pretrained(base)
  model = AutoModelForSequenceClassification.from_pretrained(base).float()
  lab = label_ids(model.config)
  if not args.separate:
    model = wrap_lora(model, r=args.r)
    model.print_trainable_parameters()
  t = time.perf_counter()
  log = train_loop(model, batches_from(rows, tok, lab, args.batch_size, args.max_length, args.seed),
                   args.max_steps, args.lr, device)
  out.mkdir(parents=True, exist_ok=True)
  model.save_pretrained(out)
  if args.separate:
    tok.save_pretrained(out)
  (out / "train_log.json").write_text(json.dumps({
    "task": args.task, "base": base, "mode": "separate" if args.separate else f"lora_r{args.r}", "rows": len(rows),
    "max_steps": args.max_steps, "batch_size": args.batch_size, "lr": args.lr, "device": device,
    "seconds": round(time.perf_counter() - t, 1), "loss": log}, indent=2))
  print(f"saved {out}")


if __name__ == "__main__":
  main()
