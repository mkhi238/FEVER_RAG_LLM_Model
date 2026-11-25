from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["TRUE", "FALSE", "NOT ENOUGH INFO"]


#Define the data that is coming into the API
#(was named ClaimsRequest here while main.py imported ClaimRequest, so the API never started)
class ClaimRequest(BaseModel):
  claim: str = Field(min_length=3, max_length=1000)  #Incoming request must have a string claim field


class Evidence(BaseModel):
  text: str
  source: str = Field(description='FEVER Wikipedia sentence id, "Page#line"')


#Defines the data that comes out of the API
class FactCheckResponse(BaseModel):
  claim: str
  prediction: Label
  response: str
  confidence: float = Field(description="Verifier's top probability; 0 when no evidence was found")
  probabilities: dict[str, float] | None = None
  evidence: list[Evidence] = []
  low_confidence: bool = Field(False, description="Verifier said TRUE/FALSE below the threshold, so NEI is returned")
  latency_ms: float
  # rear aggregator (only set when it is enabled; prediction is then its verdict)
  verifier_prediction: Label | None = Field(None, description="Verifier's label before the aggregator's veto")
  veto_requested: bool = Field(False, description="Aggregator's LLM asked to veto (may be ignored by the confidence gate)")
  vetoed: bool = Field(False, description="Veto was applied: the verifier's TRUE/FALSE became NEI")
  veto_reason: str | None = None
  llm_confidence: float | None = Field(None, description="Aggregator LLM's confidence that the claim is unfactual")
  explanation: str | None = Field(None, description="Aggregator's explanation, citing evidence as [page:line]")
  aggregator_error: str | None = Field(None, description="Aggregator failed; the verifier's verdict is returned")
