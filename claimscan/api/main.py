"""FEVER fact-check API. Run: uvicorn claimscan.api.main:app

Kept as-is apart from import fixes and on_event -> lifespan. The ClaimScan plan
replaces this with LangServe later.
"""
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException

from claimscan.api.deps import model_service
from claimscan.api.schemas import ClaimRequest, FactCheckResponse
from claimscan.evidence.select import get_evidence
from claimscan.pipeline import verify


#runs automatically when the server starts (replaces the deprecated @app.on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
  model_service.load() #loads model
  yield

app = FastAPI(title = "FEVER Fact Checker API", lifespan = lifespan)

@app.post("/fact-check", response_model = FactCheckResponse)
async def fact_check_endpoint(request: ClaimRequest):
  try:
    evidence_results = model_service.retriever.search(request.claim, k = 10)
    evidence = get_evidence(request.claim, evidence_results, model_service.corpus_df)

    if not evidence:
      return FactCheckResponse(
        claim = request.claim,
        prediction = "NOT ENOUGH INFO",
        response = "No relevant evidence found",
        confidence = 0.0
      )

    prediction, probs = verify(model_service.model, request.claim, evidence)

    confidence = torch.max(probs).item()

    if confidence < 0.55:
      prediction = "NOT ENOUGH INFO"

    if prediction == 'TRUE':
      response = f"This statement is TRUE. Evidence: {evidence[0]}"
    elif prediction == "FALSE":
      response = f"This statement is FALSE. Evidence: {evidence[0]}"
    else:
      response = "Insufficient evidence."

    return FactCheckResponse(
      claim = request.claim,
      prediction = prediction,
      response = response,
      confidence = float(confidence)
    )

  except Exception as e:
    raise HTTPException(status_code = 500, detail = str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": model_service.is_loaded()}


