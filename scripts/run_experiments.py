import mlflow
import sys
from pathlib import Path
import torch
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add paths
inference_path = Path(r'C:\Users\mukun\crisis-claim-analysis\src\rag\Inference')
verifier_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Verifier")
retriever_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Retriever")

sys.path.insert(0, str(inference_path))
sys.path.insert(0, str(verifier_dir))
sys.path.insert(0, str(retriever_dir))

# Import everything you need
from test_rag import fact_check_cleaned
from DeBERTa_Model import FEVERVerifier
from faiss_sentence_retriever import FAISSRetriever
from config import config

# Initialize the global objects that fact_check_cleaned needs
print("Loading model and retriever...")

# Load model
model = FEVERVerifier()
model_path = config.verifier_output_dir / "trained_model" / "model.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# Load retriever
retriever = FAISSRetriever(config.retriever_index_dir, config.retriever_model)

# Load corpus
corpus_df = pd.read_parquet(config.corpus)

# Make them available to test_rag module
import test_rag
test_rag.model = model
test_rag.retriever = retriever
test_rag.corpus_df = corpus_df

print("Loading complete!\n")

mlflow.set_experiment("fever_fact_checking")
print("completed")
test_claims = [
    # Clear factual claims
    ("Water boils at 100 degrees Celsius", "TRUE"),
    ("Tokyo is the capital of Japan", "TRUE"),
    ("Albert Einstein developed the theory of relativity", "TRUE"),
    
    # Current/temporal claims
    ("Trump is the president of the United States", "TRUE"),  # Former president
    ("Joe Biden is the current president of the United States", "FALSE"),
    ("Barack Obama is the former president of the United States", "TRUE"),
    ("India was also a colony of the UK", "TRUE"),
    ("Socrates was a real person", "TRUE"),
    
    # Sports/records
    ("Lebron James is the all time points leader in the NBA", "TRUE"),
    ("Michael Jordan played for the Chicago Bulls", "TRUE"),
    
    # Clear false claims
    ("Cats can fly", "FALSE"),
    ("The Pacific Ocean is the smallest ocean", "FALSE"),
    ("The Earth is flat", "FALSE"),
    ("The Eiffel Tower is in London", "FALSE"),
    ("Australia is the largest country in the world", "FALSE"),
    ("The sun orbits the Earth", "FALSE"),
    ("Steph Curry is the all time 3-pt leader in the WNBA", "FALSE"),
    
    # Authorship claims
    ("Charles Dickens wrote Oliver Twist", "TRUE"),
    ("J.K. Rowling wrote Harry Potter", "TRUE"),
    ("Hamlet is a play by Shakespeare", "TRUE"),
    ("George Orwell wrote 1984", "TRUE"),
    ("Jane Austen wrote Pride and Prejudice", "TRUE"),
    
    # Geographic claims
    ("Rome is the capital of Italy", "TRUE"),
    ("Moscow is located in Russia", "TRUE"),
    ("The Amazon rainforest is in South America", "TRUE"),
    
    # Biographical claims
    ("Steve Jobs founded Apple", "TRUE"),
    ("Martin Luther King Jr. was born in Atlanta", "TRUE"),
    ("Leonardo da Vinci painted the Mona Lisa", "TRUE"),
    ("Drake made Gods Plan", "TRUE"),
    
    # Edge cases - these are too vague to verify
    ("The weather is nice today", "NOT ENOUGH INFO"),
    ("This coffee tastes good", "NOT ENOUGH INFO")
]

def run_experiment(k_value, conf_threshold):
  with mlflow.start_run(run_name = f"k{k_value}_conf{conf_threshold}"):
    mlflow.log_params({
      "retrieval_k": k_value,
      "confidence_threshold": conf_threshold
    })

    correct = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    pred_counts = {"TRUE": 0, "FALSE": 0, "NOT ENOUGH INFO": 0}
    true_counts = {"TRUE": 0, "FALSE": 0, "NOT ENOUGH INFO": 0}

    y_true = []
    y_pred = []
    for claim, true_label in test_claims:
      pred = fact_check_cleaned(claim, k_value, conf_threshold)
      pred_counts[pred] += 1
      true_counts[true_label] += 1
      
      y_true.append(true_label)
      y_pred.append(pred)

      if pred == true_label:
        correct += 1

      if pred == "TRUE" and true_label == "TRUE":
        true_positives += 1
      elif pred == "TRUE" and true_label != pred:
        false_positives += 1
      elif pred != true_label and true_label == "TRUE":
        false_negatives += 1
      
    accuracy = correct/len(test_claims)
    recall = true_positives / (false_negatives + true_positives) #more important, since we need to know if true claims are bing caught
    precision = true_positives / (true_positives + false_positives)
    f1_score = (2 * precision * recall) / (precision + recall)

    mlflow.log_metrics({
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1-score": f1_score
    })

    print(f"k={k_value}, threshold={conf_threshold}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1-Score: {f1_score:.1%}")

    labels = ["TRUE", "FALSE", "NOT ENOUGH INFO"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (k={k_value}, threshold={conf_threshold})')
    plt.tight_layout()

    cm_path = f"confusion_matrix_k{k_value}_conf{conf_threshold}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()  # Important: free memory

run_experiment(k_value=10, conf_threshold=0.50)
run_experiment(k_value=30, conf_threshold=0.65)
run_experiment(k_value=30, conf_threshold=0.50)
run_experiment(k_value=50, conf_threshold=0.50)

