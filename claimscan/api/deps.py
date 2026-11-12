"""Holds the loaded models for the API (was dependancy.py; main.py imported api.dependencies)."""
from claimscan.config import config
from claimscan.pipeline import load_services


class ModelService:
  def __init__(self):
    self.model = None
    self.retriever = None
    self.corpus_df = None

  def load(self):
    #loads verifier, retriever and corpus into memory
    services = load_services(config)
    self.model = services.model
    self.retriever = services.retriever
    self.corpus_df = services.corpus_df

    print('Load Complete')

  def is_loaded(self):
    return self.model is not None

#load model into RAM
model_service = ModelService()
