"""Build verifier training data: (claim, evidence sentences, label) examples.

TRUE / FALSE examples use the claim's gold evidence; NOT ENOUGH INFO examples use the
claim's top FAISS hits that are not gold evidence. Up to ``--per-class`` of each.

Fixes over the first version (whose output trained the current checkpoint):
  * line 0: gold line 0 is stored as -1 (eval/gold.py); it was never matched, so ~46% of
    gold sentences were lost and gold line-0 sentences could be used as NEI negatives.
  * speed: sentences are looked up in an (id, line)-indexed corpus and evidence is grouped
    by claim once, instead of scanning 25M corpus rows / 367k evidence rows per claim.
  * repeated gold sentences (same sentence in several annotation sets) are kept once.
  * ``--split validation`` builds a held-out set from FEVER's dev claims.

Usage:
  python -m claimscan.verifier.prepare                       # train split -> training_data_v2.json
  python -m claimscan.verifier.prepare --split validation --per-class 500 --out verifier_val.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from claimscan.config import config
from claimscan.eval.gold import fix_line, load_claims, norm_title
from claimscan.evidence.select import _lookup_text, index_corpus

LABELS = ("TRUE", "FALSE", "NOT ENOUGH INFO")


def group_gold_pairs(evidence_df):
  """{claim id: [(page, line), ...]} in file order, line-0 fixed, no duplicates. NEI rows (no page) are skipped."""
  ev = evidence_df[evidence_df["evidence_wiki_url"].notna() & (evidence_df["evidence_wiki_url"] != "")]
  out = {}
  for cid, page, line in zip(ev["id"], ev["evidence_wiki_url"], ev["evidence_sentence_id"]):
    pairs = out.setdefault(int(cid), [])
    pair = (norm_title(page), fix_line(line))
    if pair not in pairs:
      pairs.append(pair)
  return out


def get_evidence_text(gold_pairs, corpus):
  """Texts of the gold (page, line) pairs found in the corpus."""
  texts = (_lookup_text(corpus, page, line) for page, line in gold_pairs)
  return [t for t in texts if t is not None]


def get_negative_text(claim_text, gold_pairs, corpus, retriever, k=20):
  """Texts of the top-k FAISS hits that are not gold evidence, up to negative_samples_per_claim."""
  gold = set(gold_pairs)
  negatives = []
  for result in retriever.search(claim_text, k=k):
    pair = (norm_title(result["id"]), int(result["line"]))
    if pair in gold:
      continue
    text = _lookup_text(corpus, *pair)
    if text is not None:
      negatives.append(text)
      if len(negatives) >= config.negative_samples_per_claim:
        break
  return negatives


def create_training_data(claims_df, gold_by_claim, corpus, retriever, per_class=2500, n_claims=75000, seed=19):
  claims = claims_df.sample(n=min(n_claims, len(claims_df)), random_state=seed)
  found = {label: [] for label in LABELS}

  def add(label, claim_id, claim_text, sentences):
    found[label].append({
      "claim_id": claim_id,
      "claim": claim_text,
      "evidence": sentences[:config.max_evidence_sentences],
      "label": label,
    })

  for row in tqdm(claims.itertuples(index=False), total=len(claims), desc="Processing claims"):
    claim_id, claim_text, label = int(row.id), str(row.claim), str(row.label_text)
    gold_pairs = gold_by_claim.get(claim_id, [])

    if gold_pairs and label in ("TRUE", "FALSE") and len(found[label]) < per_class:
      sentences = get_evidence_text(gold_pairs, corpus)
      if sentences:
        add(label, claim_id, claim_text, sentences)

    if len(found["NOT ENOUGH INFO"]) < per_class:
      try:
        negatives = get_negative_text(claim_text, gold_pairs, corpus, retriever)
      except Exception as e:  # one bad query shouldn't kill a multi-hour run
        print(f"Retrieval failed for claim {claim_id}: {e}")
        negatives = []
      if negatives:
        add("NOT ENOUGH INFO", claim_id, claim_text, negatives)

    if all(len(v) >= per_class for v in found.values()):
      break

  print("\nCollected: " + ", ".join(f"{k}={len(v)}" for k, v in found.items()))
  return [ex for label in LABELS for ex in found[label]]


def main():
  default_out = config.verifier_training_data.with_name("training_data_v2.json")
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--split", default="train", help="train / validation / test")
  ap.add_argument("--per-class", type=int, default=2500)
  ap.add_argument("--n-claims", type=int, default=75000, help="claims sampled from the split")
  ap.add_argument("--out", type=Path, default=default_out)
  args = ap.parse_args()

  from claimscan.retrieval.faiss_retriever import FAISSRetriever

  claims_df = load_claims(args.split)
  gold_by_claim = group_gold_pairs(pd.read_parquet(config.evidence))
  print("Loading and indexing the corpus (a few minutes, ~several GB RAM)...")
  corpus = index_corpus(pd.read_parquet(config.corpus, columns=["id", "line", "text"]))
  retriever = FAISSRetriever(config.retriever_index_dir, config.retriever_model)

  examples = create_training_data(claims_df, gold_by_claim, corpus, retriever, args.per_class, args.n_claims)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  with open(args.out, "w") as f:
    json.dump(examples, f, indent=2)
  print(f"Saved {len(examples)} examples to {args.out}")


if __name__ == "__main__":
  main()
