from pathlib import Path

class VerifierConfig():
  def __init__(self):
    
    self.claims_file = Path(r"D:\crisis-claim-analysis\data\processed\fever_claims_parquet")
    self.gold_file = Path(r"D:\crisis-claim-analysis\data\processed\fever_gold.json")
    self.corpus = Path(r"D:\crisis-claim-analysis\data\processed\fever_corpus_parquet")
    self.evidence = Path(r"D:\crisis-claim-analysis\data\processed\fever_evidence_parquet")

    self.retriever_index_dir = Path(r"D:\crisis-claim-analysis\artifacts\dense\faiss_bge_small_en_v1_5")
    self.retriever_index_dir.mkdir(parents = True, exist_ok = True)
    self.retriever_model = "BAAI/bge-small-en-v1.5"

    self.verifier_model = "microsoft/deberta-v3-base"
    self.verifier_output_dir = Path(r"D:\crisis-claim-analysis\artifacts\verifier\deberta_v1")
    self.verifier_output_dir.mkdir(parents=True, exist_ok=True)

    self.batch_size: int = 8
    self.learning_rate: float = 2e-5
    self.num_epochs: int = 3
    self.max_length: int = 512
    self.warmup_steps: int = 500

    self.negative_samples_per_claim: int = 1
    self.max_evidence_sentences: int = 3      

    self.eval_batch_size = 32
    #YES, NO, NEI
    self.num_labels = 3
    

    self.label_mapping = {
          "TRUE": 0,
          "FALSE": 1,
          "NOT ENOUGH INFO": 2
        }

    self.mlflow_experiment_name = "fever_verifier"
    self.mlflow_tracking_uri = str(Path(r"D:\crisis-claim-analysis\artifacts\mlruns"))
    
config = VerifierConfig()

def validate_config():
  required_files = [
    config.claims_file,
    config.gold_file,
    config.corpus,
    config.evidence
  ]

  required_dirs = [
    config.retriever_index_dir
  ]

  missing_files = [f for f in required_files if not f.exists()]
  missing_dirs = [d for d in required_dirs if not d.exists()]
    
  if missing_files:
      raise FileNotFoundError(f"Missing required files: {missing_files}")
    
  if missing_dirs:
      raise FileNotFoundError(f"Missing required directories: {missing_dirs}")
  
  else:
     print("Configuration Complete")
                                                          
if __name__ == "__main__":
    validate_config()