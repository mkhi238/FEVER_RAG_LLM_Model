from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import trl.kernels.chunked_logprob as chunked

import torch
import yaml

from claimscan.data.posttrain_store import load_sft_dataset, load_dpo_dataset
from claimscan.instruct.chat_clients import ROLES_FILE

with open(ROLES_FILE) as f:
  cfg = yaml.safe_load(f)["roles"]["aggregator"]   # the rear aggregator's role in llamacpp/roles.yaml

# VitaminC train + SufficientFacts from posttrain.sqlite (python -m claimscan.data.posttrain_store)

#SFT
dataset_sft = load_sft_dataset()

chunked._MM_OUT_DTYPE = False   # RTX 2060: TRL's loss kernel otherwise needs an Ampere GPU

args = SFTConfig(
    output_dir="/mnt/d/crisis-claim-analysis/artifacts/models/aggregator_sft",   # on D:
    model_init_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager",   # bf16: Gemma overflows in fp16
                       "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                 bnb_4bit_compute_dtype=torch.bfloat16)},   # 4-bit: fits 6 GB
    per_device_train_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True,
    max_length=1024, max_steps=400, logging_steps=10, report_to="none",
)

trainer = SFTTrainer(
    model=cfg.get("hf_model"),
    train_dataset=dataset_sft,
    args=args,
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),   # PEFT: train LoRA adapters only
)
for p in trainer.model.parameters():   # LoRA weights in bf16 too
  if p.requires_grad:
    p.data = p.data.to(torch.bfloat16)
trainer.train()
trainer.save_model()


#DPO
dataset_dpo = load_dpo_dataset()

sft_model = trainer.model   # start DPO from the SFT model (4-bit base + SFT LoRA), not the raw base
del trainer
torch.cuda.empty_cache()   # free the SFT optimizer state

args_dpo = DPOConfig(
    output_dir="/mnt/d/crisis-claim-analysis/artifacts/models/aggregator_dpo",   # on D:
    per_device_train_batch_size=1, gradient_accumulation_steps=8, gradient_checkpointing=True,
    max_length=1024, max_steps=150, logging_steps=10, report_to="none",
)

trainer = DPOTrainer(
    model=sft_model,   # PEFT model: the reference is the same model with the adapter off, so no 2nd copy in VRAM
    train_dataset=dataset_dpo,
    args=args_dpo,
)
trainer.train()
trainer.save_model()






