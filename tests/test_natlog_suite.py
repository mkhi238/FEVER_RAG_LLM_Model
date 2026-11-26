"""Natural-logic claim suite: retrieve -> rerank -> the natural-logic verifier, on N FEVER validation claims.

The full-model test for the natural-logic layer (docs/OVERHAUL.md). It skips until
claimscan/neurosymbolic_layer/natlog/verify.py exists, whose entry point is

    verify_claim(claim: str, evidence: list[str], nli, encode) -> NatlogResult
      .label   "TRUE" | "FALSE" | "NOT ENOUGH INFO"
      .proof   [(claim_span, evidence_span, operator), ...]
      .used    indices into evidence of the sentences that matter
      .enough  is the evidence enough

Like tests/test_fever_suite.py it needs the artifacts on D: (cached hits, sentence store), so a plain `pytest`
deselects it (tests/conftest.py); run it by naming the file:

    pytest tests/test_natlog_suite.py                      # 10 claims (default)
    pytest tests/test_natlog_suite.py --natlog-claims 300  # the whole sample

Same claims and retrieval as the FEVER suite (its cached hybrid hits, a label-balanced slice of the 300-claim
sample), the same reranker and k as the pipeline, so the numbers sit next to the FEVER suite's. ``nli`` is the
config's DeBERTa NLI verifier, ``encode`` the retriever's bge-small encoder. Scored only with the existing FEVER
metric (claimscan/eval/fever_eval.py: label accuracy, FEVER score, evidence precision / recall / F1), with the
predicted evidence = the ``used`` sentences. Results go to <eval_dir>/natlog_suite/latest_n<N>.json.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from test_fever_suite import BASELINE_FILE, LABELS, _services_factory, suite_hits

pytestmark = pytest.mark.natlog_suite


@pytest.fixture(scope="module")
def natlog_run(request):
  natlog = pytest.importorskip("claimscan.neurosymbolic_layer.natlog.verify")
  pytest.importorskip("pandas")
  from claimscan.config import config as cfg
  from claimscan.eval import benchmark as B
  from claimscan.eval.fever_eval import fever_scores, to_prediction, to_reference
  from claimscan.evidence.select import _lookup_text
  from claimscan.pipeline import _rerank

  n = request.config.getoption("--natlog-claims")
  base = json.loads(BASELINE_FILE.read_text())
  if not 1 <= n <= len(base["predictions"]):
    pytest.fail(f"--natlog-claims must be 1..{len(base['predictions'])}, got {n}")
  for p in (cfg.sentence_store,):
    if not Path(p).exists():
      pytest.skip(f"artifact missing: {p}")
  claims, fused, all_fused, gold, index_tag = suite_hits(cfg, B, base, n)
  services = _services_factory(cfg, B, all_fused, base, Path(cfg.eval_dir) / "fever_suite",
                               cfg.verifier_inference_model, index_tag)(None)
  from sentence_transformers import SentenceTransformer
  encode = SentenceTransformer(cfg.retriever_model, device="cpu").encode

  t = time.perf_counter()
  preds, refs, rows = [], [], []
  for c in claims.itertuples():
    cid = int(c.id)
    hits = fused[cid][:cfg.rerank_candidates if services.reranker else base["k"]]
    if services.reranker is not None:
      hits = _rerank(services.reranker, c.claim, hits, services.corpus_df, base["k"])
    found = [(f"{h['id']}#{h['line']}", _lookup_text(services.corpus_df, h["id"], h["line"])) for h in hits]
    found = [(d, s) for d, s in found if s is not None]
    docids, evidence = [d for d, _ in found], [s for _, s in found]
    r = natlog.verify_claim(c.claim, evidence, services.model, encode)
    used = [docids[i] for i in r.used]
    preds.append(to_prediction(r.label, used))
    refs.append(to_reference(c.label_text, gold.get(cid, [])))
    rows.append({"id": cid, "claim": c.claim, "gold": c.label_text, "label": r.label, "enough": r.enough,
                 "used": used, "proof": [list(p) for p in r.proof], "n_evidence": len(evidence),
                 "used_idx": list(r.used)})

  result = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_claims": n, "retriever": "hybrid_norm3",
            "reranker": cfg.rerank_backend, "nli": cfg.verifier_inference_model, "encoder": cfg.retriever_model,
            "labels": claims["label_text"].value_counts().to_dict(), "e2e": fever_scores(preds, refs),
            "wall_clock_s": round(time.perf_counter() - t, 1), "per_claim": rows}
  out_dir = Path(cfg.eval_dir) / "natlog_suite"
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / f"latest_n{n}.json").write_text(json.dumps(result, indent=2))
  print(f"\nnatlog on {n} claims: " + "  ".join(f"{k}={v:.3f}" for k, v in result["e2e"].items()))
  return result


def test_natlog_outputs_are_well_formed(natlog_run):
  """The verifier's outputs have the planned shape on every claim. The scores are in latest_n<N>.json and printed;
  there is no baseline to compare them against yet."""
  problems = []
  for r in natlog_run["per_claim"]:
    if r["label"] not in LABELS:
      problems.append(f"{r['id']}: label {r['label']!r}")
    if not isinstance(r["enough"], bool):
      problems.append(f"{r['id']}: enough {r['enough']!r} is not a bool")
    if any(not 0 <= i < r["n_evidence"] for i in r["used_idx"]):
      problems.append(f"{r['id']}: used {r['used_idx']} outside the {r['n_evidence']} evidence sentences")
    if any(len(p) != 3 for p in r["proof"]):
      problems.append(f"{r['id']}: proof steps must be (claim_span, evidence_span, operator)")
  assert not problems, "\n".join(problems)
  assert set(natlog_run["e2e"]) >= {"label_accuracy", "fever_score", "evidence_precision", "evidence_recall",
                                    "evidence_f1"}
