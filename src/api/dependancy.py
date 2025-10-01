import torch
from pathlib import Path
import pandas as pd
import sys
verifier_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Verifier")
retriever_dir = Path(r"C:\Users\mukun\crisis-claim-analysis\src\rag\Retriever")
#sys.path is Python's list of directories to search for modules when you do import
#sys.path.insert(0, str(retriever_dir)) adds your retriever directory to the beginning of that list (position 0 = first priority)
sys.path.insert(0, str(retriever_dir))
sys.path.insert(0, str(verifier_dir))
from DeBERTa_Model import FEVERVerifier
from faiss_sentence_retriever import FAISSRetriever
from config import config

#dependancy creates a container class which holds and manages the models; sets up a manager to access and load them


class ModelService:
  def __init__(self):
    self.model = None
    self.retriever = None
    self.corpus_df = None

  def load(self):
    #loads model into memory
    self.model = FEVERVerifier()
    model_path = config.verifier_output_dir / "trained_model" / "model.pt"
    self.model.load_state_dict(torch.load(model_path))
    self.model.eval()

    #loads retriever and corpus
    self.retriever = FAISSRetriever(config.retriever_index_dir, config.retriever_model)
    self.corpus_df = pd.read_parquet(config.corpus)

    print('Load Complete')

  def is_loaded(self):
    return self.model is not None

#load model into RAM
model_service = ModelService()