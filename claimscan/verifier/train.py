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
from config import config
from DeBERTa_Model import FEVERVerifier

#the Dataset class is another framework class like nn.Module, but this class needs you to define everything since it does not come with any inherret defined methods
#The purpose of Dataset is just to allow Pytorch to know that the DatasetLoader is infact a dataset loader, meaning that the class must contain the appropriate functions

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

#batching math:
# 5000 max examples (so like maybe 7335 including positive and negatives)
#80% for training = 7335 training examples 
#Batch size 8 = 6112 ÷ 8 =  iterations per epoch (764), each batch triggers one weight updates
# Fwd pass: process 8 examples, get preds, calcualte loss, backwards pass (loss.backward(), calculate gradients for all parameters), 
# update weights (optimizer.step(), adjust all model weights using those gradients), Reset: optimizer.zero_grad() - clear gradients for next batch
# So with 764 batches per epoch, model gets updated 764 times per epoch, Over 3 epochs = 764 × 3 = 2,292 total weight updates, Each update uses gradients from 8 training examples
# Smaller batches = more frequent updates = more learning opportunities, The learning happens incrementally - the model gets slightly better after each batch as it adjusts 
# its weights based on the errors it made on those 8 examples.
#3 epochs to avoid overfit

def train_model():
  train_file = config.verifier_output_dir / "training_data.json" / "training_data.json"

  with open(train_file, 'r') as f:
    examples = json.load(f)

  
  print(f"Loaded {len(examples)} examples")

  idx = int(0.8 * len(examples))
  train_examples = examples[:idx]
  val_examples = examples[idx:]

  print(f"Training: {len(train_examples)}, Validation: {len(val_examples)}")

  model = FEVERVerifier()
  #Dataset defines how to access individual data points. It must have __len__ and __getitem__, and handles specific data format and processing
  train_dataset = FEVERDatasetLoader(train_examples, model)
  val_dataset = FEVERDatasetLoader(val_examples, model)

  #Dataloader Handles the common aspects of ALL training (batching, shuffling, etc.)
  #PyTorch cannot handle our specific JSON format, so it needs a custom Dataset class. But once done, it can efficiently batch and shuffle with DataLoader
  
  #shuffle true for training to randomize order of examples in each epoch to prevent memorization and generalizes
  #want to keep validation non-shuffled to replicate it everytime

  #training: Smaller batches for training, Each gradient update uses 8 examples,ore frequent weight updates = more learning opportunities and less GPU memory
  #validation: Larger batches for validation, No gradient computation needed (using torch.no_grad()), can process more GPU memory

  #Dataloader calls the __getitem__ method automatically; when we iterate over Dataloader in 
  #for batch in train_loader
  #Dataloader will automatically call the daataset, and will then do 
  #for idx in index: train_datset.__getitem__(idx) 
  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  #the .to method is another inherited method from nn.module, moves the model to either CPU or GPU memory
  model.to(device)
  optimizer = AdamW(model.parameters(), lr = config.learning_rate, weight_decay=0.01)
  criterion = nn.CrossEntropyLoss()

  train_losses, train_accs, val_accs = [], [], []
  for epoch in range(config.num_epochs):
    #set it into training mode
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc = f"Epoch {epoch+1} Training"):
      #able to get these since train_loader automatically runs __getitem__()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)

      #this step resets the gradients once they have been updated, so new gradients arent cumulative and new gradients are adjusted without the old ones
      optimizer.zero_grad() # Reset gradients

      #forward pass
      logits = model(input_ids, attention_mask)

      #Crossentropy loss being run
      loss = criterion(logits, labels)

      #backwards pass/backpropogation (.backwards() is a nn method that can be run on 2d tensors only)
      # Calculate gradients  
      loss.backward() #calculates the gradients, but doenst apply them 

      # Update weights
      optimizer.step() #applies the gradients

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
  
  epochs = np.arange(1, len(train_losses) + 1) #+1 needed since range doesnt include stop val
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
  train_file = config.verifier_output_dir / "training_data.json"

  if not train_file.exists():
    print(f"Training data not found: {train_file}")

  else:
    model = train_model()
    print("Complete")