"""Glue between fact_check_detailed and the rear aggregator (claimscan/instruct/rear_aggregator.py).

The aggregator (an LLM on llama-server, or a hosted OpenAI-compatible API) reads the verifier's verdict + evidence and may veto it,
which can only turn TRUE/FALSE into NOT ENOUGH INFO. The veto is applied only when the verifier is
unsure (conf < 0.6) or the LLM is sure (llm_confidence >= 0.9); that gate is fixed in
rear_aggregator.final_label, not configured here. It is optional and FAILS OPEN: if the server
is down, slow or returns garbage, the verifier's verdict is returned unchanged with
``aggregator_error`` set.

langchain and claimscan.instruct are imported lazily, so this module (and the API) loads without
langchain installed while the aggregator is disabled.
"""
from __future__ import annotations

import inspect
import logging
import os
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from claimscan.pipeline import Verdict

log = logging.getLogger("claimscan.aggregation")

NEI = "NOT ENOUGH INFO"
LABELS = ("TRUE", "FALSE", NEI)
MAX_EVIDENCE = 5     # FEVER convention: only the first 5 predicted sentences count


class VetoOutput(BaseModel):
  """What the structured LLM returns per claim. Fallback only: build_llm uses the instruct package's VetoOutput when it has one."""
  veto: bool = Field(description="True if the evidence does not actually support the verdict")
  veto_reason: str = Field(description="Why the verdict was vetoed; empty if not vetoed")
  llm_confidence: float = Field(ge=0, le=1, description="How sure the model is that the claim is unfactual or impossible")
  explanation: str = Field(description="1-3 sentences, citing evidence as [page:line]")


class UnavailableLLM:
  """Stands in for the LLM when it can't be built (e.g. the API key env var is missing): every claim fails open."""
  def __init__(self, error: str):
    self.error = error

  def __repr__(self):
    return f"UnavailableLLM({self.error!r})"


_secrets: set[str] = set()   # key values in use, scrubbed from log lines


def _redact(text: str) -> str:
  for s in _secrets:
    text = text.replace(s, "***")
  return re.sub(r"sk-[A-Za-z0-9*_\-.]{4,}", "sk-***", text)   # provider errors echo partial keys


@dataclass
class AggregatedVerdict:
  verdict: Verdict                    # the verifier's result, unchanged
  label: str                          # final label (the aggregator's verdict, or the verifier's on failure)
  verifier_label: str                 # verdict.label, before the veto
  veto_requested: bool = False        # the LLM asked for a veto
  vetoed: bool = False                # the veto was applied (passed the confidence gate)
  veto_reason: str | None = None
  llm_confidence: float | None = None
  explanation: str | None = None
  aggregator_error: str | None = None  # set when the aggregator failed and the verifier label was kept


def build_llm(cfg):
  """The structured runnable aggregate() expects, pointed at cfg.aggregator_base_url. Needs langchain_openai.

  The API key comes from the env var NAMED by cfg.aggregator_api_key_env (empty = local llama-server, no key).
  If that var is unset/empty, returns an UnavailableLLM so every request fails open with that reason.
  """
  key_env = cfg.aggregator_api_key_env
  if key_env:
    api_key = os.environ.get(key_env, "")
    if not api_key:
      log.warning("rear aggregator: API key env var %s is missing or empty; every request will fail open", key_env)
      return UnavailableLLM(f"API key env var {key_env} is missing or empty")
    _secrets.add(api_key)
  else:
    api_key = "not-needed"   # llama-server ignores the key

  from claimscan.instruct import model, rear_aggregator, schemas
  # the instruct package's VetoOutput wins; the local class is only a fallback until it defines one
  schema = getattr(rear_aggregator, "VetoOutput", None) or getattr(schemas, "VetoOutput", None) or VetoOutput
  kw = dict(base_url=cfg.aggregator_base_url, timeout=cfg.aggregator_timeout)
  if "api_key" in inspect.signature(model.build_chat).parameters:
    kw["api_key"] = api_key
  try:
    if key_env and "api_key" not in kw:
      raise NotImplementedError   # build_chat can't take a key yet: a hosted API needs the inline client
    chat = model.build_chat("aggregator", **kw)
  except NotImplementedError:
    # fallback until claimscan/instruct/model.py build_chat is written; delete it then
    from langchain_openai import ChatOpenAI
    chat = ChatOpenAI(base_url=cfg.aggregator_base_url, model=cfg.aggregator_model, api_key=api_key,
                      timeout=cfg.aggregator_timeout, max_retries=0, temperature=0)
  return chat.with_structured_output(schema, method="json_schema")


def _split_docid(docid: str) -> tuple[str, int]:
  page, _, line = docid.rpartition("#")
  return page, int(line)


def to_claim_result(claim: str, v: Verdict):
  """fact_check_detailed's Verdict -> instruct ClaimResult (top-5 evidence).

  label is v.label, i.e. after the confidence threshold, so a downgraded verdict reaches the
  aggregator as NEI and the veto can never bring it back.
  """
  from claimscan.instruct.schemas import ClaimResult, Evidence
  evidence = []
  for text, docid in v.evidence[:MAX_EVIDENCE]:
    page, line = _split_docid(docid)
    evidence.append(Evidence(page=page, line=line, text=text))
  return ClaimResult(claim=claim, label=v.label, probs={l: v.probs[l] for l in LABELS}, evidence=evidence)


def aggregate_verdict(claim: str, v: Verdict, llm) -> AggregatedVerdict:
  """Run the rear aggregator on one claim's verdict. Never raises."""
  base = AggregatedVerdict(verdict=v, label=v.label, verifier_label=v.label)
  if llm is None or v.probs is None:
    # disabled, or no evidence was found (already NEI, nothing to veto)
    return base
  if isinstance(llm, UnavailableLLM):
    base.aggregator_error = f"aggregator unavailable ({llm.error})"
    return base
  try:
    from langchain_core.runnables import RunnableLambda

    from claimscan.instruct.rear_aggregator import aggregate
    from claimscan.instruct.schemas import RearInput
    raw = []   # what the LLM itself returned, so an ignored veto stays visible
    recorded = llm | RunnableLambda(lambda o: raw.append(o) or o)
    out = aggregate(RearInput(user_text=claim, results=[to_claim_result(claim, v)]), recorded)
    c = out.claims[0]
    # the veto may only downgrade to NEI; anything else (e.g. a flip or "UNCLEAR") is treated as garbage
    if c.verdict not in (v.label, NEI):
      raise ValueError(f"aggregator returned {c.verdict!r} for verifier label {v.label!r}")
    vetoed = c.verdict != v.label
    requested = bool(getattr(raw[0], "veto", vetoed)) if raw else vetoed
    # prefer the fields aggregate() copied onto the verdict, else the LLM's own output
    first = raw[0] if raw else None
    llm_conf = getattr(c, "llm_confidence", None)
    if llm_conf is None:
      llm_conf = getattr(first, "llm_confidence", None)
    reason = getattr(c, "veto_reason", None) or getattr(first, "veto_reason", None) or None
    return AggregatedVerdict(verdict=v, label=c.verdict, verifier_label=v.label, veto_requested=requested,
                             vetoed=vetoed, veto_reason=reason,
                             llm_confidence=llm_conf, explanation=c.explanation or None)
  except Exception as e:
    log.warning("rear aggregator failed for claim %r, keeping the verifier verdict: %s", claim,
                _redact(f"{type(e).__name__}: {e}"))
    base.aggregator_error = f"aggregator unavailable ({type(e).__name__})"
    return base
