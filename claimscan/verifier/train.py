import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from claimscan.config import config
from claimscan.verifier.model import FEVERVerifier

class FEVERDatasetLoader(Dataset):
  def __init__(self, examples, model):
    self.examples = examples
    self.model = model

  def __len__(self):
    return len(self.examples)
  
  def __getitem__(self, idx):
    example = self.examples[idx]

    claim = example['claim']
    evidence = example['evidence']
    label = config.label_mapping[example['label']]

    encoding = self.model.encode_text(claim, evidence)

    return {
      'input_ids': encoding['input_ids'].squeeze(),
      'attention_mask': encoding['attention_mask'].squeeze(),
      'label': torch.tensor(label, dtype=torch.long)

    } 

def train_model():
  train_file = config.verifier_training_data

  with open(train_file, 'r') as f:
    examples = json.load(f)

  
  print(f"Loaded {len(examples)} examples")

  idx = int(0.8 * len(examples))
  train_examples = examples[:idx]
  val_examples = examples[idx:]

  print(f"Training: {len(train_examples)}, Validation: {len(val_examples)}")

  model = FEVERVerifier()
  train_dataset = FEVERDatasetLoader(train_examples, model)
  val_dataset = FEVERDatasetLoader(val_examples, model)

  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  model.to(device)
  optimizer = AdamW(model.parameters(), lr = config.learning_rate, weight_decay=0.01)
  criterion = nn.CrossEntropyLoss()

  train_losses, train_accs, val_accs = [], [], []
  for epoch in range(config.num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc = f"Epoch {epoch+1} Training"):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      optimizer.zero_grad() # Reset gradients

      #forward pass
      logits = model(input_ids, attention_mask)

      #Crossentropy loss being run
      loss = criterion(logits, labels)

      # Calculate gradients  
      loss.backward()

      # Update weights
      optimizer.step()

      total_loss += loss.item()
      preds = torch.argmax(logits, dim = 1)
      correct += (preds == labels).sum().item()
      total += labels.size(0)

    train_acc = correct/total
    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(input_ids, attention_mask)
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    val_acc = correct/total

    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"  Train Acc: {train_acc:.4f}")
    print(f"  Val Acc: {val_acc:.4f}")
  
  epochs = np.arange(1, len(train_losses) + 1)
  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_losses, 'b-', label='Training Loss')
  plt.title('Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)

  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
  plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)

  plt.tight_layout()
  plt.show()

  save_dir = config.verifier_output_dir / "trained_model"
  save_dir.mkdir(exist_ok=True, parents=True)
  torch.save(model.state_dict(), save_dir / "model.pt")
  print(f"Model saved to {save_dir}")

  return model

if __name__ == "__main__":
  train_file = config.verifier_training_data

  if not train_file.exists():
    print(f"Training data not found: {train_file}")

  else:
    model = train_model()
    print("Complete")