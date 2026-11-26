from trl import SFTTrainer

import yaml

from claimscan.data.posttrain_store import load_sft_dataset
from claimscan.instruct.chat_clients import ROLES_FILE

with open(ROLES_FILE) as f:
  cfg = yaml.safe_load(f)["roles"]["aggregator"]   # the rear aggregator's role in llamacpp/roles.yaml

# VitaminC train + SufficientFacts from posttrain.sqlite (python -m claimscan.data.posttrain_store)

#SFT
dataset = load_sft_dataset()

trainer = SFTTrainer(
    model=cfg.get("hf_model"),   
    train_dataset=dataset,
)
trainer.train()

#DPO
