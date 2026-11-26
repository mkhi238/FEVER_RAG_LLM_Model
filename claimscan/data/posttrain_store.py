"""Post-training data in one SQLite table, `sft_examples`, and the TRL loader posttraining.py uses.

    python -m claimscan.data.posttrain_store context    # VitaminC, HoVer, LIAR2, UltraFeedback (minutes)
    python -m claimscan.data.posttrain_store verify     # aggregator rows: VitaminC + SufficientFacts through the verifier (CPU)
    python -m claimscan.data.posttrain_store fever      # aggregator rows: FEVER train claims through the full pipeline (CPU, ~3 h)

sft_examples (source, ref, system, user, assistant): one finished chat example per row. Sources:
  agg_*          the rear aggregator's task: its exact prompt (rear_aggregator.build_messages) -> VetoOutput JSON.
                 Veto when the verifier's TRUE/FALSE is wrong. agg_fever = FEVER train claims the verifier never
                 trained on (never validation, the eval set); agg_vitaminc / agg_sf_* = external evidence, verifier only.
  vitaminc       claim + evidence -> TRUE / FALSE / NOT ENOUGH INFO
  hover          multi-hop claim + its evidence from our Wikipedia SQLite -> SUPPORTED / NOT SUPPORTED
  liar           statement + fact-checker summary -> one of LIAR's 6 ratings
  ultrafeedback  general instructions -> the preferred answer (alignment)
Each builder replaces its own sources, except fever, which resumes. Downloads go to the HF cache (on D:).
"""
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from claimscan.config import config

ROOT = Path(config.artifacts_root) / "data" / "posttrain"
DB = ROOT / "posttrain.sqlite"
HOVER = ROOT / "hover" / "hover_train_release_v1.1.json"   # github.com/hover-nlp/hover data/hover/
LABELS = {"SUPPORTS": "TRUE", "SUPPORTED": "TRUE", "REFUTES": "FALSE", "NOT ENOUGH INFO": "NOT ENOUGH INFO"}
LIAR_LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]   # label 0..5
NEI = "NOT ENOUGH INFO"

VITAMINC_PROMPT = """You get a claim and evidence. Answer TRUE if the evidence supports the claim, FALSE if it
refutes it, or NOT ENOUGH INFO if it does neither. Answer with the label only."""
HOVER_PROMPT = """You get a claim and Wikipedia sentences, tagged [page:line]. The claim may need several sentences
together. Answer SUPPORTED if the sentences together support every part of the claim, otherwise NOT SUPPORTED.
Answer with the label only."""
LIAR_PROMPT = """You get a political statement, who said it, and a fact-checker's summary. Rate the statement as
pants-fire, false, barely-true, half-true, mostly-true or true. Answer with the rating only."""


# ---------- helpers ----------

def _hf(repo, file):
  from huggingface_hub import hf_hub_download
  return pd.read_parquet(hf_hub_download(repo, file, repo_type="dataset", revision="refs/convert/parquet"))


def _connect():
  con = sqlite3.connect(DB)
  con.execute("CREATE TABLE IF NOT EXISTS sft_examples (source TEXT, ref TEXT, system TEXT, user TEXT, assistant TEXT)")
  return con


def _replace(con, source, rows):
  """rows: (ref, system, user, assistant). Replaces everything from `source`."""
  con.execute("DELETE FROM sft_examples WHERE source = ?", (source,))
  con.executemany("INSERT INTO sft_examples VALUES (?, ?, ?, ?, ?)", ((source, *r) for r in rows))
  con.commit()
  print(f"{source}: {len(rows):,} rows")


# ---------- context + alignment ----------

def build_vitaminc(con, n):
  d = _hf("tals/vitaminc", "default/train/0000.parquet")
  d = d.sample(n=min(n, len(d)), random_state=0)
  _replace(con, "vitaminc", [(r.unique_id, VITAMINC_PROMPT, f"Claim: {r.claim}\nEvidence: {r.evidence}", LABELS[r.label])
                             for r in d.itertuples(index=False)])


def _fever_id(title):
  """HoVer title -> our Wikipedia page id (FEVER style)."""
  for a, b in {" ": "_", "(": "-LRB-", ")": "-RRB-", ":": "-COLON-", "[": "-LSB-", "]": "-RSB-"}.items():
    title = title.replace(a, b)
  return title


def build_hover(con, n):
  """Evidence looked up in fever_sentences.sqlite (~98% found); claims with any sentence missing are skipped."""
  from claimscan.evidence.store import SentenceStore
  store = SentenceStore(config.sentence_store)
  with open(HOVER) as f:
    claims = pd.DataFrame(json.load(f)).sample(frac=1, random_state=0)
  rows = []
  for r in claims.itertuples(index=False):
    lines = []
    for title, line in r.supporting_facts:
      text = store.lookup(_fever_id(title), line)
      if text is None:
        break
      lines.append(f"[{_fever_id(title)}:{line}] {text}")
    else:
      label = "SUPPORTED" if r.label == "SUPPORTED" else "NOT SUPPORTED"
      rows.append((r.uid, HOVER_PROMPT, f"Claim: {r.claim}\nEvidence:\n" + "\n".join(lines), label))
      if len(rows) == n:
        break
  _replace(con, "hover", rows)


def build_liar(con, n):
  d = _hf("chengxuphd/liar2", "default/train/0000.parquet").dropna(subset=["justification"])
  d = d.sample(n=min(n, len(d)), random_state=0)
  _replace(con, "liar", [(str(r.id), LIAR_PROMPT,
                          f"Statement: {r.statement}\nSpeaker: {r.speaker}\nFact-checker summary: {r.justification}",
                          LIAR_LABELS[int(r.label)]) for r in d.itertuples(index=False)])


def build_ultrafeedback(con, n, max_chars=3000):
  """Single-turn instructions and the preferred answer; long answers are skipped (they won't fit in training)."""
  d = _hf("HuggingFaceH4/ultrafeedback_binarized", "default/train_sft/0000.parquet").sample(frac=1, random_state=0)
  rows = []
  for r in d.itertuples(index=False):
    m = list(r.messages)
    if len(m) == 2 and m[0]["role"] == "user" and m[1]["role"] == "assistant" and len(m[1]["content"]) <= max_chars:
      rows.append((r.prompt_id, None, m[0]["content"], m[1]["content"]))
      if len(rows) == n:
        break
  _replace(con, "ultrafeedback", rows)


# ---------- aggregator ----------

class _Given:
  """Retriever that returns a dataset's own evidence, so fact_check_detailed runs unchanged on it."""
  def __init__(self, hits):
    self.hits = hits

  def search(self, query, k=30):
    return self.hits[:k]


def _agg_row(ref, claim, verdict, gold):
  """A pipeline Verdict -> (ref, system, user, assistant) for the aggregator, or None if there was no evidence
  (the API never asks the aggregator then). Target: veto when the verifier's TRUE/FALSE is wrong."""
  from claimscan.aggregation import to_claim_result
  from claimscan.instruct.rear_aggregator import build_messages
  if verdict.probs is None:
    return None
  result = to_claim_result(claim, verdict)
  veto = result.label != NEI and result.label != gold
  cite = f"[{result.evidence[0].page}:{result.evidence[0].line}]" if result.evidence else "the evidence"
  answer = {
    "veto": veto,
    "veto_reason": f"The evidence does not support the verdict {result.label}." if veto else "",
    "explanation": (f"The verifier said {result.label}, but {cite} does not back that up." if veto
                    else f"{cite} is consistent with the verdict {result.label}."),
    "llm_confidence": 0.95 if veto else 0.05,   # >= 0.9 so final_label applies the veto even when the verifier is sure
  }
  system, user = build_messages(result)
  return (str(ref), system.content, user.content, json.dumps(answer))


def build_verified(con, n):
  """Verifier only, on each external example's given evidence (same decision logic as the API)."""
  from claimscan.pipeline import Services, fact_check_detailed, load_verifier
  model = load_verifier()
  vc = _hf("tals/vitaminc", "default/train/0000.parquet")
  sets = {"agg_vitaminc": [(r.unique_id, r.claim, [(r.page, r.evidence)], LABELS[r.label])
                           for r in vc.sample(n=min(n, len(vc)), random_state=1).itertuples(index=False)]}
  for s in ("hover", "vitaminc"):
    sf = _hf("copenlu/sufficient_facts", f"{s}/test/0000.parquet")
    # label_after "NOT ENOUGH": the removed evidence was needed -> NEI; "ENOUGH -- ...": the original label stands
    gold = sf.label_before.map(LABELS).where(sf.label_after != "NOT ENOUGH", NEI)
    sets[f"agg_sf_{s}"] = [(i, c, [(str(t), str(x)) for t, x in ev], g)
                           for i, c, ev, g in zip(sf.index, sf.claim, sf.evidence, gold)][:n]
  for source, examples in sets.items():
    rows = []
    for ref, claim, pairs, gold in examples:
      corpus = pd.DataFrame([(p, j, t) for j, (p, t) in enumerate(pairs)], columns=["id", "line", "text"])
      hits = [{"id": p, "line": j, "score": 1.0} for j, (p, _) in enumerate(pairs)]
      row = _agg_row(ref, claim, fact_check_detailed(claim, Services(model, _Given(hits), corpus)), gold)
      if row:
        rows.append(row)
    _replace(con, source, rows)


def build_fever(con, n):
  """FEVER train claims the verifier never trained on, through the full pipeline. Resumes where it stopped."""
  from claimscan.eval.gold import load_claims
  from claimscan.pipeline import fact_check_detailed, load_services
  with open(config.verifier_training_data) as f:
    seen = {str(r["claim_id"]) for r in json.load(f)}
  claims = load_claims("train")
  claims = claims[~claims["id"].astype(str).isin(seen)].sample(n=n, random_state=0)
  done = {r[0] for r in con.execute("SELECT ref FROM sft_examples WHERE source = 'agg_fever'")}
  services = load_services()
  for i, r in enumerate(claims.itertuples(index=False), 1):
    if str(r.id) not in done:
      row = _agg_row(r.id, r.claim, fact_check_detailed(r.claim, services), r.label_text)
      if row:
        con.execute("INSERT INTO sft_examples VALUES ('agg_fever', ?, ?, ?, ?)", row)
    if i % 50 == 0:
      con.commit()
      print(f"agg_fever: {i}/{n}", flush=True)
  con.commit()


# ---------- TRL dataset ----------

def load_sft_dataset(limit=None):
  """Every row as {"prompt": [system?, user], "completion": [assistant]}, shuffled. limit = first N rows."""
  from datasets import Dataset

  with sqlite3.connect(DB) as con:
    df = pd.read_sql_query("SELECT * FROM sft_examples ORDER BY RANDOM()", con)
  if limit:
    df = df.head(limit)

  rows = []
  for system, user, assistant in zip(df.system, df.user, df.assistant):
    prompt = [{"role": "system", "content": system}] if pd.notna(system) and system else []   # NULL comes back as NaN
    rows.append({
      "prompt": prompt + [{"role": "user", "content": user}],
      "completion": [{"role": "assistant", "content": assistant}],
    })
  return Dataset.from_list(rows)


if __name__ == "__main__":
  cmd = sys.argv[1:2]
  with _connect() as con:
    if cmd == ["context"]:
      build_vitaminc(con, 5000)
      build_hover(con, 5000)
      build_liar(con, 5000)
      build_ultrafeedback(con, 5000)
    elif cmd == ["verify"]:
      build_verified(con, 2000)
    elif cmd == ["fever"]:
      build_fever(con, 5000)
    else:
      print(__doc__)
