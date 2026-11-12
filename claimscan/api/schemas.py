from pydantic import BaseModel
#Pydantic models define the structure and types of data your API expects and returns.

#Define the data that is coming into the API
class ClaimsRequest(BaseModel):
  claim : str #Incoming request mucst have a string claim field
#FastAPI will validate checks if "claim" field exists, check if it is a str, reject if validation fails or pass Python object if it passes


#Defines the data that comes out of the API
#this data will be in the JSON form
#FastAPI validates incoming data (rejects bad requests)
#Document your API automatically (showing the format to use)
#Type-check your code (preventing bugs) 
class FactCheckResponse(BaseModel):
  claim: str 
  prediction: str
  response: str
  confidence: float

#Pydantic automatically checks if requests have the right type