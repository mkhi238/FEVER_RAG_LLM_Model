import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from claimscan.config import config
from claimscan.verifier.deberta_verifier import FEVERVerifier

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

def split_examples(examples, val_frac=0.2, seed=19):
  """Stratified train/val split that keeps all of a claim's examples on one side.

  prepare.py writes examples grouped by label (all TRUE, then FALSE, then NEI), so the old
  ``examples[:80%]`` split gave a validation set that was 100% NEI. Claims are grouped by
  claim_id (a claim can have both a TRUE and an NEI example) and stratified by their label set.
  """
  by_claim = defaultdict(list)
  for ex in examples:
    by_claim[ex.get("claim_id", id(ex))].append(ex)
  strata = defaultdict(list)
  for cid, exs in by_claim.items():
    strata[tuple(sorted({e["label"] for e in exs}))].append(cid)

  rng = random.Random(seed)
  val_ids = set()
  for key in sorted(strata):
    cids = sorted(strata[key], key=str)
    rng.shuffle(cids)
    val_ids.update(cids[:round(len(cids) * val_frac)])

  train = [e for cid, exs in by_claim.items() if cid not in val_ids for e in exs]
  val = [e for cid, exs in by_claim.items() if cid in val_ids for e in exs]
  rng.shuffle(train)
  return train, val


def label_counts(examples):
  counts = defaultdict(int)
  for e in examples:
    counts[e["label"]] += 1
  return dict(counts)


def evaluate(model, loader, device, desc="Validation"):
  """Accuracy, per-class recall and the confusion matrix (rows = gold, cols = predicted)."""
  model.eval()
  preds, golds = [], []
  with torch.no_grad():
    for batch in tqdm(loader, desc=desc):
      logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
      preds.extend(torch.argmax(logits, dim=1).tolist())
      golds.extend(batch['label'].tolist())

  names = sorted(config.label_mapping, key=config.label_mapping.get)
  cm = confusion_matrix(golds, preds, labels=list(range(len(names))))
  per_class = {n: (cm[i, i] / cm[i].sum() if cm[i].sum() else float("nan")) for i, n in enumerate(names)}
  return {"acc": float(np.trace(cm) / max(cm.sum(), 1)), "per_class": per_class, "confusion": cm, "names": names}


def print_eval(result):
  print(f"  Val Acc: {result['acc']:.4f}")
  for n, r in result["per_class"].items():
    print(f"    {n:>16}: {r:.4f}")
  print("  Confusion (rows gold, cols pred: " + ", ".join(result["names"]) + ")")
  print("\n".join("    " + str(row) for row in result["confusion"].tolist()))


def load_examples(path):
  with open(path, 'r') as f:
    return json.load(f)


def eval_checkpoint(data_file, checkpoint=None):
  """Score a saved verifier on every example in data_file (use a held-out file, e.g. from --split validation)."""
  examples = load_examples(data_file)
  print(f"Evaluating on {len(examples)} examples {label_counts(examples)}")
  model = FEVERVerifier()
  model.load_state_dict(torch.load(checkpoint or config.verifier_checkpoint, map_location="cpu"))
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  loader = DataLoader(FEVERDatasetLoader(examples, model), batch_size=config.eval_batch_size, shuffle=False)
  result = evaluate(model, loader, device, desc="Eval")
  print_eval(result)
  return result


def train_model(train_file=None, out_dir=None):
  examples = load_examples(train_file or config.verifier_training_data)
  print(f"Loaded {len(examples)} examples")

  train_examples, val_examples = split_examples(examples)
  print(f"Training: {len(train_examples)} {label_counts(train_examples)}")
  print(f"Validation: {len(val_examples)} {label_counts(val_examples)}")

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

    result = evaluate(model, val_loader, device, desc=f"Epoch {epoch+1} Validation")
    val_acc = result["acc"]

    train_losses.append(avg_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {avg_loss:.4f}")
    print(f"  Train Acc: {train_acc:.4f}")
    print_eval(result)
  
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
  save_dir = Path(out_dir or config.verifier_output_dir) / "trained_model"
  save_dir.mkdir(exist_ok=True, parents=True)
  plt.savefig(save_dir / "training_curves.png")  # plt.show() blocks / does nothing headless
  plt.close()

  torch.save(model.state_dict(), save_dir / "model.pt")
  print(f"Model saved to {save_dir}")

  return model

if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="Train the DeBERTa verifier, or --eval-only a saved checkpoint.")
  ap.add_argument("--data", type=Path, default=config.verifier_training_data, help="examples json from prepare.py")
  ap.add_argument("--eval-only", action="store_true", help="score the saved checkpoint on all of --data")
  ap.add_argument("--checkpoint", type=Path, default=None, help="default: config.verifier_checkpoint")
  ap.add_argument("--out-dir", type=Path, default=None,
                  help="save to <out-dir>/trained_model (default: config.verifier_output_dir, overwrites the current model)")
  args = ap.parse_args()

  if not args.data.exists():
    print(f"Data not found: {args.data}")
  elif args.eval_only:
    eval_checkpoint(args.data, args.checkpoint)
  else:
    train_model(args.data, args.out_dir)
