"""Load the trained DeBERTa verifier and score one claim/evidence pair.

No FAISS needed, so this runs on a laptop. Based on DeBERTa_Model.py's __main__,
but loads the trained checkpoint instead of the untrained head.
"""
import torch

from claimscan.config import config
from claimscan.pipeline import LABEL_MAP, load_verifier

if __name__ == "__main__":
  print(f"Checkpoint: {config.verifier_checkpoint}")
  model = load_verifier()

  pairs = [
    ("Barack Obama was born in Hawaii", ["Barack Obama was born In Honolulu, Hawaii."]),
    ("The Eiffel Tower is in London", ["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."]),
    ("Socrates was a real person", ["The Pacific Ocean is the largest ocean on Earth."]),
  ]
  for claim, evidence in pairs:
    encoding = model.encode_text(claim, evidence)
    with torch.no_grad():
      probs = torch.softmax(model(encoding["input_ids"], encoding["attention_mask"]), dim=1)[0]
    pred = LABEL_MAP[int(torch.argmax(probs))]
    print(f"{claim!r}: {pred}  " + "  ".join(f"{LABEL_MAP[i]}={p:.3f}" for i, p in enumerate(probs.tolist())))
