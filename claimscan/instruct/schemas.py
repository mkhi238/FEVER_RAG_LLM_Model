"""Pydantic shapes passed between the front, the verifier loop and the rear.

Flow: user text -> FrontOutput (front_decomposer) -> one ClaimResult per claim (verifier loop)
-> RearInput -> RearOutput (rear_aggregator). Use with llm.with_structured_output(Model).
"""
from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["TRUE", "FALSE", "NOT ENOUGH INFO"]   # same names as LABEL_MAP in claimscan/pipeline.py


# ---- Front ----
class FrontOutput(BaseModel):
  """What front_decomposer returns: a reply to the user plus the claims to check."""
  reply: str = Field(description="Short message to the user, e.g. 'Checking 3 claims...'")
  claims: list[str] = Field(description="Atomic, self-contained, checkable claims")


# ---- Verifier loop -> rear ----
class Evidence(BaseModel):
  """One sentence the verifier saw, with where it came from."""
  page: str = Field(description="Wikipedia page id, e.g. 'Barack_Obama'")
  line: int = Field(description="Sentence number on that page")
  text: str


class ClaimResult(BaseModel):
  """Verifier output for one claim."""
  claim: str
  label: Label                                   # argmax of probs
  probs: dict[Label, float]                      # softmax, e.g. {"TRUE": 0.91, "FALSE": 0.06, "NOT ENOUGH INFO": 0.03}
  evidence: list[Evidence]                       # exactly the top-k sentences the verifier saw, in order


class RearInput(BaseModel):
  """Everything the rear needs to write the final answer."""
  user_text: str                                 # the original input, for context
  results: list[ClaimResult]


# ---- Rear ----
class ClaimVerdict(BaseModel):
  """The rear's explanation for one claim."""
  claim: str
  verdict: Label | Literal["UNCLEAR"]            # UNCLEAR when no label is clearly ahead
  confidence: float = Field(ge=0, le=1)          # usually the top prob from the verifier
  explanation: str = Field(description="1-3 sentences, citing evidence as [page:line]")


class RearOutput(BaseModel):
  """What rear_aggregator returns to the user."""
  summary: str = Field(description="Overall answer in plain language")
  claims: list[ClaimVerdict]
