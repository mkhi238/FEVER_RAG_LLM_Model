import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from claimscan.config import config


class FEVERVerifier(nn.Module):
  def __init__(self):
    super().__init__()

    self.tokenizer = AutoTokenizer.from_pretrained(config.verifier_model)
    # .float(): transformers>=5 loads in the stored dtype (fp16 for deberta-v3-base); the
    # verifier was trained and saved in fp32 under transformers 4.x, so keep it fp32.
    self.deberta = AutoModel.from_pretrained(config.verifier_model).float()
    self.classifier = nn.Linear(self.deberta.config.hidden_size, config.num_labels)

  def forward(self, input_ids, attention_mask):
    outputs = self.deberta(input_ids = input_ids, attention_mask = attention_mask)
    cls_output = outputs.last_hidden_state[:,0]
    logits = self.classifier(cls_output)
    return logits
  
  def encode_text(self, claim, evidence):
    if evidence:
      evidence_text = " [SEP] ".join(evidence)
    else:
      evidence_text = ""
    combined = f"{claim} [SEP] {evidence_text}"

    encoding = self.tokenizer(combined,max_length = config.max_length,padding = "max_length",truncation = True,return_tensors = 'pt')

    return encoding
  
if __name__ == '__main__':
  model = FEVERVerifier()
  claim = "Barack Obama was born in Hawaii"
  evidence = ["Barack Obama was born In Honolulu, Hawaii."]
  #since model.encode_text  is a method of the model class, we can call it here
  encoding = model.encode_text(claim, evidence)
  print(f"Input shape: {encoding['input_ids'].shape}")

  with torch.no_grad():
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    print(f"Output shape: {logits.shape}")
    print(f"Predictions: {torch.softmax(logits, dim=1)}")

