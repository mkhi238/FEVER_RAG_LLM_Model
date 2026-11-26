"""QLoRA SFT of the rear aggregator (roles.yaml -> aggregator.hf_model) on posttrain.sqlite. Fits a 6 GB GPU.

    python -m claimscan.instruct.posttraining                    # train, then score the held-out agg_* rows
    python -m claimscan.instruct.posttraining --eval-only        # score the saved adapter only
    python -m claimscan.instruct.posttraining --eval-only --base # score the untrained base model

Data: python -m claimscan.data.posttrain_store (VitaminC, HoVer, LIAR, UltraFeedback, agg_*). The adapter goes to
<artifacts>/models/aggregator_sft. DPO comes later.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from claimscan.config import config
from claimscan.data.posttrain_store import load_sft_dataset
from claimscan.instruct.chat_clients import ROLES_FILE

OUT = Path(config.artifacts_root) / "models" / "aggregator_sft"


def load_model(name, adapter=None):
  from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
  quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
  model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map={"": 0},
                                               dtype=torch.bfloat16, attn_implementation="eager")
  tok = AutoTokenizer.from_pretrained(name)
  if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)
  return model, tok


def train(name, args):
  from peft import LoraConfig
  from trl import SFTConfig, SFTTrainer

  if torch.cuda.get_device_capability(0) < (8, 0):
    # TRL's fused LM head does bf16-in/fp32-out GEMMs (Ampere+); this flag makes it upcast to fp32 first instead
    import trl.kernels.chunked_logprob as chunked
    chunked._MM_OUT_DTYPE = False
  model, tok = load_model(name)
  data = load_sft_dataset(per_source=args.per_source)
  print(f"train rows: {len(data):,}", flush=True)
  cfg = SFTConfig(
    output_dir=str(OUT), max_length=args.max_length, per_device_train_batch_size=1, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=20, num_train_epochs=1, max_steps=args.max_steps,
    gradient_checkpointing=True, fp16=False, bf16=False, optim="paged_adamw_8bit", logging_steps=10, save_steps=100,
    save_total_limit=2, report_to="none", completion_only_loss=True,
  )
  lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
  trainer = SFTTrainer(model=model, args=cfg, train_dataset=data, processing_class=tok, peft_config=lora)
  # All bf16, no AMP: Gemma overflows in fp16 (loss 0 / NaN), the RTX 2060 (sm_75) has no native bf16 so torch
  # emulates it, and fp32 LoRA weights would need bf16-in/fp32-out GEMMs (Ampere only).
  for p in trainer.model.parameters():
    if p.requires_grad:
      p.data = p.data.to(torch.bfloat16)
  trainer.train(resume_from_checkpoint=args.resume or None)
  trainer.save_model(str(OUT))
  tok.save_pretrained(str(OUT))
  del trainer, model
  torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(name, adapter, n):
  """Held-out agg_* rows: share of answers that parse as VetoOutput JSON, and veto accuracy against the target."""
  model, tok = load_model(name, adapter)
  model.eval()
  rows = load_sft_dataset(split="eval", limit=n)
  ok = right = 0
  for r in rows:
    ids = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(0)
    out = model.generate(**ids, max_new_tokens=200, do_sample=False)
    text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    gold = json.loads(r["completion"][0]["content"])["veto"]
    try:
      ans = json.loads(text)
      ok += 1
      right += bool(ans.get("veto")) == gold
    except (json.JSONDecodeError, AttributeError):
      pass
  res = {"model": name, "adapter": str(adapter) if adapter else None, "n": len(rows),
         "valid_json": ok / len(rows), "veto_accuracy": right / len(rows),
         "always_no_veto": sum(not json.loads(r["completion"][0]["content"])["veto"] for r in rows) / len(rows)}
  print(json.dumps(res), flush=True)
  with open(OUT.parent / "aggregator_sft_eval.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--eval-only", action="store_true")
  p.add_argument("--base", action="store_true", help="with --eval-only: score the model without the adapter")
  p.add_argument("--per-source", type=int, default=1000, help="rows per non-aggregator source (agg_* all kept)")
  p.add_argument("--max-length", type=int, default=1024)
  p.add_argument("--max-steps", type=int, default=-1)
  p.add_argument("--eval-n", type=int, default=150)
  p.add_argument("--resume", default="")
  args = p.parse_args()
  with open(ROLES_FILE) as f:
    name = yaml.safe_load(f)["roles"]["aggregator"]["hf_model"]
  OUT.mkdir(parents=True, exist_ok=True)
  if not args.eval_only:
    train(name, args)
  evaluate(name, None if args.base else OUT, args.eval_n)


if __name__ == "__main__":
  main()
