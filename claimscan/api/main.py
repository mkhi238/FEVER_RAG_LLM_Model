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

#Creates your FastAPI application. The title appears in the auto-generated docs at /docs
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



#async means the function can be paused and resumed, allowing other work to happen during waiting periods.
#normal functions are syncrobnous, meaning that the program stops and waits for the function to finish
#async def get_data() - When it hits async, the function pauses and lets other tasks run. When syncronous finishes, it resumes.
#async enables concurrency - ability to handle multiple things at once by switching tasks during wait time
#/fact-check endpoint being async means while one request waits for model inference, another request can start processing
#async means a function can pause and let other work happen while waiting. So like two functions would run concurrently
#async needs the await keyword - here async does nothing other than acts as a keyword for FastAPI since there is no await

#when you use async with FastAPI without await, FastAPI uses thread pools (not the async event loop) to achieve concurrency.
#FastAPI handles the threading automatically. Works well for most ML inference APIs. async just acts as a wrapper, and then allows FastAPI to
#use thread pools to work
