"""FOL verification of one claim: decomposition -> unit hypotheses -> DeBERTa per unit -> logic -> verdict + reason.

  verify_fol(claim, evidence, checker, decomposition=None, quantifier_mode="unit") -> FOLVerdict
    checker        check.VerifierChecker (the pipeline's Verify decision per unit) or check.NatlogChecker
    decomposition  from the front decomposer (claimscan/instruct/front_decomposer.py); None = the whole claim as one
                   atom (decomposition.fallback), which gives the plain verifier's label
  build_checker(model, cfg, encode=None)  the checker named by natlog.checker in default.yaml
"""
from __future__ import annotations

from claimscan.neurosymbolic_layer.fol.check import NatlogChecker, VerifierChecker
from claimscan.neurosymbolic_layer.fol.compose import FOLVerdict, compose
from claimscan.neurosymbolic_layer.fol.decomposition import Decomposition, fallback


def _cfg(cfg):
  if cfg is None:
    from claimscan.config import config
    return config
  return cfg


def settings(cfg=None) -> dict:
  """natlog: in default.yaml, with defaults for anything missing."""
  s = dict(getattr(_cfg(cfg), "natlog", None) or {})
  s.setdefault("checker", "verifier")
  s.setdefault("quantifier_mode", "unit")
  s.setdefault("min_sim", 0.6)
  s.setdefault("forward_supports", True)
  s.setdefault("decomposer_attempts", 2)
  s["tasks"] = s.get("tasks") or {}
  return s


def build_checker(model, cfg=None, encode=None, task: str = "verify"):
  """model: an HFNLIVerifier-like model (encode_text / forward) for the verifier checker, or that / an
  nli(premise, hypothesis) callable for the natlog one."""
  cfg = _cfg(cfg)
  s = settings(cfg)
  if s["checker"] == "verifier":
    return VerifierChecker(model, 0.5, getattr(cfg, "verifier_nei_prob_threshold", None))
  if s["checker"] == "natlog":
    if encode is None:
      raise ValueError("natlog.checker: natlog needs encode (the retriever's bge-small encoder)")
    per_task = (s["tasks"].get(task) or {}).get("min_sim")
    return NatlogChecker(model, encode, float(s["min_sim"] if per_task is None else per_task),
                         bool(s["forward_supports"]))
  raise ValueError(f"natlog.checker must be verifier or natlog, got {s['checker']!r}")


def verify_fol(claim: str, evidence: list[str], checker, decomposition: Decomposition | None = None,
               quantifier_mode: str = "unit") -> FOLVerdict:
  d = decomposition or fallback(claim)
  return compose(d.formula, evidence, checker, d.entities, quantifier_mode)
