import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys

#DeBERTa Model - starts pre-trained on general language understanding, gets fine tuned during training
#Learns to better represent claim-evidence representations

#Linear classifier starts with random weights and gets fully trained during training from Transformers representation

sys.path.append(str(Path(__file__).parent))
from config import config


class FEVERVerifier(nn.Module):
  def __init__(self):
    super().__init__()

    #Converts text → token IDs that the model can understand
    #Load the tokenizer that converts text → numbers that DeBERTa can process.
    self.tokenizer = AutoTokenizer.from_pretrained(config.verifier_model)
    #Processes those token IDs → contextualized representations
    #Loads the pre-trained DeBERTa model (the transformer that understands language). This gives you 768-dimensional representations of text.
    self.deberta = AutoModel.from_pretrained(config.verifier_model)
    #Takes DeBERTa's output → final prediction
    #Creates a simple linear layer that converts DeBERTa's 768-dim output → 3 classes (TRUE/FALSE/NOT ENOUGH INFO).
    self.classifier = nn.Linear(self.deberta.config.hidden_size, config.num_labels)

  def forward(self, input_ids, attention_mask):
    #Runs tokenized text through DeBERTa, gets contextualized representations for every token.
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
  #model automatically calls the forward method
  #The model ONLY does forward pass:
  #Takes inputs → runs through layers → returns outputs
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


#Note on nn.Module
#class Module:
#    def __call__(self, *args):
#        return self.forward(*args)  # this is automatically imported  
#    def forward(self):
#        raise NotImplementedError  # I need to implement forward
#    def parameters(self):
#        # Automatically finds all your layers (already implemented function)
#    def train(self):
#        # Sets all layers to training mode (already implemented function)

#nn.Module is a framework importer, but you would need to define forward always for this to work