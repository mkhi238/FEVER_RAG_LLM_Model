"""SFT data for the decomposer (Qwen3-4B): NL -> the decomposer JSON, as chat records. Data prep only, no training.

  python -m claimscan.neurosymbolic_layer.fol.sft_data                 # MALLS + FOLIO -> <artifacts>/natlog/fol_sft/
  python -m claimscan.neurosymbolic_layer.fol.sft_data --limit 500     # small

Sources (HF hub, cached under HF_HOME on D:):
  MALLS v0.1 (yuan-yang/MALLS-v0)   ~27k NL-FOL sentence pairs (GPT-4 generated, partly verified); train / test files
  FOLIO v2 (tasksource/folio)        expert-written stories: each premise line + the conclusion with its FOL line
Each FOL string goes through formula.parse (FOLIO / MALLS syntax; ↔ ⊕ rewritten); unparseable ones are dropped and
counted in stats.json. Constants become entities (id = the constant, name = its words). These sources have no
per-atom text and no queries, so the root gets the sentence as its text and queries stay empty; the per-atom texts the
prompt asks for come from the FEVER silver records.
FEVER silver (silver_records): a teacher model (any front_decomposer client, e.g. a hosted large model) decomposes
FEVER train claims; a record is kept only if it validates and fol_reward says its composed verdict on the gold
evidence equals the gold label (rejection sampling with the same reward GRPO would use). Needs a teacher and the
verifier, so main() does not run it (the teacher is the user's choice; not run yet).
Output: {train,dev}.jsonl with {"messages": [system, user, assistant], "source"}; stats.json.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from claimscan.instruct.prompts import DECOMPOSER_SYSTEM
from claimscan.neurosymbolic_layer.fol.formula import And, Atom, FOLError, Implies, Not, Or, atoms, parse, to_json
from claimscan.neurosymbolic_layer.fol.hypothesis import words

MALLS = ("yuan-yang/MALLS-v0", ("MALLS-v0.1-train.json", "MALLS-v0.1-test.json"))
FOLIO = ("tasksource/folio", ("folio_v2_train.jsonl", "folio_v2_validation.jsonl"))


def _with_text(f, text: str):
  """The same node with its text set."""
  if isinstance(f, Atom):
    return Atom(f.pred, f.args, text)
  if isinstance(f, Not):
    return Not(f.arg, text)
  if isinstance(f, (And, Or)):
    return type(f)(f.args, text)
  if isinstance(f, Implies):
    return Implies(f.left, f.right, text)
  return type(f)(f.var, f.body, text)


def to_target(nl: str, fol: str) -> dict:
  """NL sentence + FOL string -> the decomposer JSON for it (FOLError if the FOL does not parse)."""
  f = _with_text(parse(fol), nl.strip())
  consts = sorted({a for at in atoms(f) for a in at.args if not (len(a) == 1 and a.islower())})   # not variables
  return {"claims": [{"claim": nl.strip(), "entities": {c: words(c) for c in consts}, "fol": to_json(f),
                      "queries": []}]}


def record(nl: str, target: dict, source: str) -> dict:
  return {"messages": [{"role": "system", "content": DECOMPOSER_SYSTEM}, {"role": "user", "content": nl.strip()},
                       {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)}], "source": source}


def malls_pairs(items) -> list[tuple[str, str, str]]:
  return [(d["NL"], d["FOL"], "malls") for d in items if d.get("NL") and d.get("FOL")]


def folio_pairs(lines) -> list[tuple[str, str, str]]:
  out = []
  for line in lines:
    d = json.loads(line)
    nls = [s for s in d.get("premises", "").split("\n") if s.strip()]
    fols = [s for s in d.get("premises-FOL", "").split("\n") if s.strip()]
    if len(nls) == len(fols):
      out += [(n, f, f"folio:{d.get('story_id')}") for n, f in zip(nls, fols)]
    if d.get("conclusion") and d.get("conclusion-FOL"):
      out.append((d["conclusion"], d["conclusion-FOL"], f"folio:{d.get('story_id')}"))
  return out


def convert(pairs) -> tuple[list[dict], dict]:
  """(records, stats). Duplicate sentences are kept once."""
  recs, seen, stats = [], set(), {"pairs": 0, "kept": 0, "duplicate": 0, "dropped": {}}
  for nl, fol, source in pairs:
    stats["pairs"] += 1
    if nl.strip() in seen:
      stats["duplicate"] += 1
      continue
    try:
      target = to_target(nl, fol)
    except FOLError as e:
      msg = str(e)
      key = ("equality / unsupported symbol" if msg.startswith("cannot read") else
             "unbound variable" if "not bound" in msg else re.sub(r"'[^']*'", "'…'", msg)[:60])
      stats["dropped"][key] = stats["dropped"].get(key, 0) + 1
      continue
    seen.add(nl.strip())
    recs.append(record(nl, target, source))
    stats["kept"] += 1
  return recs, stats


def silver_records(items, client, checker, attempts: int = 2) -> list[dict]:
  """FEVER silver: items = [(claim, gold_evidence_sentences, gold_label)]; keep the teacher's decomposition only
  when it is valid and its composed verdict matches the gold label."""
  from claimscan.instruct.front_decomposer import decompose
  from claimscan.neurosymbolic_layer.fol.reward import fol_reward
  out = []
  for claim, evidence, gold in items:
    r = decompose(claim, client, attempts)
    if r.fallback:
      continue
    target = {"claims": [d.to_json() for d in r.claims]}
    if fol_reward(json.dumps(target), evidence, gold, checker).label == gold:
      out.append(record(claim, target, "fever_silver"))
  return out


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--limit", type=int, default=None, help="max records per source")
  ap.add_argument("--dev-frac", type=float, default=0.05)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)
  from huggingface_hub import hf_hub_download

  from claimscan.config import config as cfg
  from claimscan.neurosymbolic_layer.fol.verify import settings
  out = Path(cfg.artifacts_root) / settings(cfg).get("fol_sft_dir", "natlog/fol_sft")
  get = lambda repo, name: hf_hub_download(repo, name, repo_type="dataset")

  all_recs, stats = [], {}
  malls = [p for name in MALLS[1] for p in malls_pairs(json.load(open(get(MALLS[0], name), encoding="utf-8")))]
  folio = [p for name in FOLIO[1] for p in folio_pairs(open(get(FOLIO[0], name), encoding="utf-8"))]
  for src, pairs in (("malls", malls), ("folio", folio)):
    recs, st = convert(pairs[:args.limit * 2] if args.limit else pairs)
    all_recs += recs[:args.limit] if args.limit else recs
    stats[src] = st
  rng = random.Random(args.seed)
  rng.shuffle(all_recs)
  n_dev = int(len(all_recs) * args.dev_frac)
  out.mkdir(parents=True, exist_ok=True)
  for name, rows in (("dev", all_recs[:n_dev]), ("train", all_recs[n_dev:])):
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
      for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
  stats["train"], stats["dev"] = len(all_recs) - n_dev, n_dev
  (out / "stats.json").write_text(json.dumps(stats, indent=2))
  print(json.dumps(stats, indent=2))
  print(f"-> {out}")


if __name__ == "__main__":
  main()
