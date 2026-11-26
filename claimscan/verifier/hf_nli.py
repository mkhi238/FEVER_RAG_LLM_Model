"""An off-the-shelf HuggingFace NLI model used as the verifier (e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

The only verifier; pipeline.verify uses this interface:
  encode_text(claim, evidence) -> encoding;  model(input_ids, attention_mask) -> logits in LABEL_MAP order
  (0 TRUE, 1 FALSE, 2 NOT ENOUGH INFO).
NLI input is a pair: premise = the evidence sentences joined, hypothesis = the claim.
Labels: entailment -> TRUE, contradiction -> FALSE, neutral -> NOT ENOUGH INFO.
Chosen with verifier.inference_model in configs/default.yaml.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_TO_FEVER = {"entailment": 0, "contradiction": 1, "neutral": 2}   # -> LABEL_MAP index


class HFNLIVerifier(nn.Module):
  def __init__(self, name: str, max_length: int = 512):
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained(name)
    self.nli = AutoModelForSequenceClassification.from_pretrained(name).float()
    self.max_length = max_length
    id2label = {int(i): l.lower() for i, l in self.nli.config.id2label.items()}
    # order[j] = the NLI output column that holds LABEL_MAP label j
    self.order = [next(i for i, l in id2label.items() if NLI_TO_FEVER[l] == j) for j in range(3)]

  def encode_text(self, claim, evidence):
    return self.tokenizer(" ".join(evidence), claim, max_length=self.max_length, truncation=True,
                          return_tensors="pt")

  def forward(self, input_ids, attention_mask):
    logits = self.nli(input_ids=input_ids, attention_mask=attention_mask).logits
    return logits[:, self.order]
