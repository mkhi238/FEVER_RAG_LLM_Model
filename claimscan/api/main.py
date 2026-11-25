"""FEVER fact-check API. Run: uvicorn claimscan.api.main:app

Serves exactly the evaluated pipeline (claimscan.pipeline.fact_check_detailed, k=30, threshold
0.5), so the benchmark numbers describe what this endpoint returns. The ClaimScan plan
replaces this with LangServe later.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from claimscan.aggregation import aggregate_verdict
from claimscan.api.deps import model_service
from claimscan.api.schemas import ClaimRequest, Evidence, FactCheckResponse
from claimscan.pipeline import fact_check_detailed

log = logging.getLogger("claimscan.api")

K = 30                 # hits retrieved per claim  (same as the benchmark / fact_check_cleaned)
CONF_THRESHOLD = 0.5   # below this a TRUE/FALSE verdict becomes NOT ENOUGH INFO


#runs automatically when the server starts (replaces the deprecated @app.on_event("startup"))
@asynccontextmanager
async def lifespan(app: FastAPI):
  model_service.load() #loads verifier + retriever, opens the sentence store
  yield

app = FastAPI(title="FEVER Fact Checker API", lifespan=lifespan)


def _response_text(label: str, evidence: list) -> str:
  if label == "TRUE":
    return f"This statement is TRUE. Evidence: {evidence[0][0]}"
  if label == "FALSE":
    return f"This statement is FALSE. Evidence: {evidence[0][0]}"
  return "Insufficient evidence."


# plain def (not async): FastAPI runs it in a worker thread, so the CPU-bound inference
# doesn't block the event loop and /health stays responsive during a request
@app.post("/fact-check", response_model=FactCheckResponse)
def fact_check_endpoint(request: ClaimRequest):
  if not model_service.is_loaded():
    raise HTTPException(status_code=503, detail="Models are still loading")
  t = time.perf_counter()
  try:
    with model_service.lock:
      v = fact_check_detailed(request.claim, model_service.services, k=K, conf_threshold=CONF_THRESHOLD)
  except Exception:
    log.exception("fact-check failed for claim %r", request.claim)
    raise HTTPException(status_code=500, detail="Internal error while checking the claim")

  # rear aggregator (off by default): may veto TRUE/FALSE to NEI; never raises (fails open)
  extra = {}
  label = v.label
  if model_service.aggregator_llm is not None:
    a = aggregate_verdict(request.claim, v, model_service.aggregator_llm)
    label = a.label
    extra = dict(verifier_prediction=a.verifier_label, veto_requested=a.veto_requested, vetoed=a.vetoed,
                 veto_reason=a.veto_reason, llm_confidence=a.llm_confidence, explanation=a.explanation,
                 aggregator_error=a.aggregator_error)

  if not v.evidence:
    response = "No relevant evidence found"
  else:
    response = extra.get("explanation") or _response_text(label, v.evidence)

  return FactCheckResponse(
    claim=request.claim,
    prediction=label,
    response=response,
    confidence=v.confidence,
    probabilities=v.probs,
    evidence=[Evidence(text=text, source=docid) for text, docid in v.evidence],
    low_confidence=v.downgraded,
    latency_ms=(time.perf_counter() - t) * 1000,
    **extra,
  )


@app.get("/health")
def health():
  return {"status": "healthy", "models_loaded": model_service.is_loaded()}
