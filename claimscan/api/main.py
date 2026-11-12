from fastapi import FastAPI, HTTPException
import torch
from api.models import ClaimRequest, FactCheckResponse
from api.dependencies import model_service
from score_evidence import get_evidence

#Creates your FastAPI application. The title appears in the auto-generated docs at /docs
app = FastAPI(title = "FEVER Fact Checker API")

#This is an event handler that runs automatically when your server starts
@app.on_event("startup") #decorator that runs the function when the API starts up
async def startup(): #async not used but convention to use
  model_service.load() #loads model

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
    encoding = model_service.model.encode_text(request.claim, evidence[:3])

    with torch.no_grad():
      logits = model_service.model(encoding['input_ids'], encoding['attention_mask'])
      prediction_idx = torch.argmax(logits, dim=1).item()
      probs = torch.softmax(logits, dim=1)[0]
    
    label_map = {0: "TRUE", 1: "FALSE", 2: "NOT ENOUGH INFO"}
    prediction = label_map[prediction_idx]

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
