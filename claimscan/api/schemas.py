from pydantic import BaseModel

#Define the data that is coming into the API
#(was named ClaimsRequest here while main.py imported ClaimRequest, so the API never started)
class ClaimRequest(BaseModel):
  claim : str #Incoming request mucst have a string claim field


#Defines the data that comes out of the API
class FactCheckResponse(BaseModel):
  claim: str
  prediction: str
  response: str
  confidence: float
