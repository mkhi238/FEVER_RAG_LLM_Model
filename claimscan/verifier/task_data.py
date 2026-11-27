"""Training pairs for the per-task NLI adapters: premise / hypothesis / label (entailment | contradiction | neutral).

  python -m claimscan.verifier.task_data --task define          # one task (or: all)
  python -m claimscan.verifier.task_data --task all --limit 2000   # small, for a smoke training run

Writes <artifacts>/<natlog.task_data_dir>/<task>/{train,dev}.jsonl. Hypotheses use the inference templates in
tasks.py, so training and use see the same wording. Sources (HF hub, cached under HF_HOME on D:):
  define    Open English WordNet 2023 (jon-tow/open-english-wordnet-synset-2023), noun synsets with a hypernym:
              premise "<term>: <gloss>", hypothesis "<term> is a <hypernym>"            -> entailment
              same premise, hypothesis "<term> is a <sibling of the hypernym>"         -> contradiction (disjoint class)
              an unrelated synset's gloss as premise, the true hypothesis               -> neutral (not in evidence)
  example   the same WordNet, instance_hypernym first (Paris -> city), else hypernym / grand-hypernym:
              "<instance> is an example of <concept>", built the same three ways
  explain   PAWS labeled_final (google-research-datasets/paws): paraphrase -> entailment, not -> neutral
              (PAWS non-paraphrases are word swaps; they are not reliably contradictions, so they are not labelled so)
  question  SQuAD 2.0 (rajpurkar/squad_v2): the gold answer -> entailment, another question's answer from the same
              context -> contradiction, an unanswerable question with some answer from its context -> neutral
  operators MED (tasksource/med) monotonicity pairs, labels as given: for the span-level operator NLI (natlog/ops.py)
  verify    none: the FEVER verifier itself is the Verify model (frozen; it is also the RL reward)
Labels are heuristic (WordNet siblings are not always disjoint, etc.); fine for adapters, documented in NATLOG_FOL.md.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from claimscan.verifier.tasks import hypothesis

DATA_TASKS = ("define", "example", "explain", "question", "operators")
WORDNET = ("jon-tow/open-english-wordnet-synset-2023", "open_english_wordnet_2023.jsonl")
PAWS = ("google-research-datasets/paws", "labeled_final/{split}-00000-of-00001.parquet")
SQUAD = ("rajpurkar/squad_v2", "squad_v2/{split}-00000-of-00001.parquet")
MED = ("tasksource/med", "data/train-00000-of-00001-ad612dfa64b78ddf.parquet")


def _hub(repo: str, filename: str) -> str:
  from huggingface_hub import hf_hub_download
  return hf_hub_download(repo, filename, repo_type="dataset")


def _row(premise, hyp, label, source):
  return {"premise": premise, "hypothesis": hyp, "label": label, "source": source}


# ---- WordNet (define, example) ----
def _lemma(member: str) -> str:
  """'oewn-rational_motive-n' -> 'rational motive'."""
  return member.split("-", 1)[1].rsplit("-", 1)[0].replace("_", " ")


def _article(phrase: str) -> str:
  return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def load_wordnet(lines) -> dict[str, dict]:
  """Noun synsets: id -> {lemma, gloss, hyper (ids), inst (instance_hypernym ids)}."""
  out = {}
  for line in lines:
    d = json.loads(line)
    if d.get("@partOfSpeech") != "n" or not d.get("Definition"):
      continue
    rels = d.get("SynsetRelation") or []
    out[d["@id"]] = {"lemma": _lemma(d["@members"].split()[0]), "gloss": d["Definition"],
                     "hyper": [r["@target"] for r in rels if r["@relType"] == "hypernym"],
                     "inst": [r["@target"] for r in rels if r["@relType"] == "instance_hypernym"]}
  return out


def wordnet_pairs(syn: dict[str, dict], task: str, rng: random.Random, limit: int | None) -> list[dict]:
  children: dict[str, list[str]] = {}
  for sid, s in syn.items():
    for h in s["hyper"]:
      children.setdefault(h, []).append(sid)
  ids = sorted(syn)
  rng.shuffle(ids)
  rows = []
  for sid in ids:
    s = syn[sid]
    if task == "example":
      ups = [h for h in s["inst"] if h in syn] or [g for h in s["hyper"] if h in syn for g in [h, *syn[h]["hyper"]]
                                                   if g in syn]
      if not s["inst"] and len(ups) < 2:      # plain nouns: only with a grand-hypernym (poodle -> dog -> canine)
        continue
    else:
      ups = [h for h in s["hyper"] if h in syn]
    if not ups:
      continue
    up = ups[0]
    grand = [g for g in syn[up]["hyper"] if g in syn]
    siblings = [c for g in grand for c in children.get(g, []) if c != up and c not in s["hyper"]]
    term = s["lemma"]
    premise = f"{term[:1].upper() + term[1:]}: {s['gloss'].rstrip('.')}."

    def hyp(concept):
      if task == "define":
        return hypothesis("define", term=term[:1].upper() + term[1:], definition=_article(concept))
      return hypothesis("example", instance=term[:1].upper() + term[1:], concept=_article(concept))

    rows.append(_row(premise, hyp(syn[up]["lemma"]), "entailment", f"wordnet:{sid}"))
    if siblings:
      rows.append(_row(premise, hyp(syn[rng.choice(siblings)]["lemma"]), "contradiction", f"wordnet:{sid}"))
    other = syn[ids[rng.randrange(len(ids))]]
    if other is not s and other["lemma"] != term:
      rows.append(_row(f"{other['lemma'][:1].upper() + other['lemma'][1:]}: {other['gloss'].rstrip('.')}.",
                       hyp(syn[up]["lemma"]), "neutral", f"wordnet:{sid}"))
    if limit and len(rows) >= limit:
      break
  return rows[:limit] if limit else rows


# ---- PAWS (explain), SQuAD 2 (question), MED (operators) ----
def paws_pairs(df, limit: int | None) -> list[dict]:
  rows = [_row(r.sentence1, hypothesis("explain", original=r.sentence2),
               "entailment" if int(r.label) == 1 else "neutral", f"paws:{r.id}") for r in df.itertuples()]
  return rows[:limit] if limit else rows


def squad_pairs(df, rng: random.Random, limit: int | None) -> list[dict]:
  rows = []
  groups = list(df.groupby("context", sort=False))
  rng.shuffle(groups)                       # SQuAD 2 is ordered by article; a --limit slice must still mix in NEI
  for context, group in groups:
    answers = [(r.question, list(r.answers["text"])) for r in group.itertuples()]
    spans = sorted({a for _, ans in answers for a in ans})
    for q, ans in answers:
      if ans:
        rows.append(_row(context, hypothesis("question", question=q, answer=ans[0]), "entailment", "squad2"))
        wrong = [s for s in spans if s not in ans and not any(s in a or a in s for a in ans)]
        if wrong:
          rows.append(_row(context, hypothesis("question", question=q, answer=rng.choice(wrong)), "contradiction",
                           "squad2"))
      elif spans:
        rows.append(_row(context, hypothesis("question", question=q, answer=rng.choice(spans)), "neutral",
                         "squad2:unanswerable"))
    if limit and len(rows) >= limit:
      break
  return rows[:limit] if limit else rows


def med_pairs(df, limit: int | None) -> list[dict]:
  ok = df[df["gold_label"].isin(["entailment", "contradiction", "neutral"])]
  rows = [_row(r.sentence1, r.sentence2, r.gold_label, f"med:{r.genre}") for r in ok.itertuples()]
  return rows[:limit] if limit else rows


def _split(rows, rng, dev_frac=0.1):
  rows = list(rows)
  rng.shuffle(rows)
  n = max(1, int(len(rows) * dev_frac)) if len(rows) > 1 else 0
  return rows[n:], rows[:n]


def build(task: str, limit: int | None = None, seed: int = 0) -> tuple[list[dict], list[dict]]:
  """(train, dev) rows for one task, from the hub files."""
  import pandas as pd
  rng = random.Random(seed)
  if task in ("define", "example"):
    with open(_hub(*WORDNET), encoding="utf-8") as f:
      syn = load_wordnet(f)
    return _split(wordnet_pairs(syn, task, rng, limit), rng)
  if task == "explain":
    train = paws_pairs(pd.read_parquet(_hub(PAWS[0], PAWS[1].format(split="train"))), limit)
    dev = paws_pairs(pd.read_parquet(_hub(PAWS[0], PAWS[1].format(split="validation"))), limit and max(limit // 10, 1))
    return train, dev
  if task == "question":
    train = squad_pairs(pd.read_parquet(_hub(SQUAD[0], SQUAD[1].format(split="train"))), rng, limit)
    dev = squad_pairs(pd.read_parquet(_hub(SQUAD[0], SQUAD[1].format(split="validation"))), rng,
                      limit and max(limit // 10, 1))
    return train, dev
  if task == "operators":
    return _split(med_pairs(pd.read_parquet(_hub(*MED)), limit), rng)
  raise ValueError(f"no training data for task {task!r} (choose from {DATA_TASKS}; verify uses the FEVER verifier)")


def write(rows: list[dict], path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for r in rows:
      f.write(json.dumps(r, ensure_ascii=False) + "\n")


def task_data_dir(cfg=None) -> Path:
  from claimscan.neurosymbolic_layer.fol.verify import _cfg, settings
  cfg = _cfg(cfg)
  return Path(cfg.artifacts_root) / settings(cfg).get("task_data_dir", "natlog/task_data")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--task", default="all", choices=("all",) + DATA_TASKS)
  ap.add_argument("--limit", type=int, default=None, help="max training rows per task (default: all)")
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args(argv)
  out = task_data_dir()
  for task in DATA_TASKS if args.task == "all" else (args.task,):
    train, dev = build(task, args.limit, args.seed)
    write(train, out / task / "train.jsonl")
    write(dev, out / task / "dev.jsonl")
    counts = {lab: sum(r["label"] == lab for r in train) for lab in ("entailment", "contradiction", "neutral")}
    print(f"{task}: {len(train)} train / {len(dev)} dev -> {out / task}  {counts}")


if __name__ == "__main__":
  main()
