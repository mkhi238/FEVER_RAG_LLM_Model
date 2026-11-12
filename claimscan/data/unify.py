import pandas as pd
from pathlib import Path
from claimscan.config import config
from claimscan.data.common import normalize_text, build_gold_map
from claimscan.data.fever_claim import build_fever_claim
from claimscan.data.fever_evidence import build_fever_evidence
from claimscan.data.fever_wiki_dump import build_corpus

path = config.data_processed

def main():
  path.mkdir(parents=True, exist_ok=True)
  claims_df = build_fever_claim()
  evidence_df = build_fever_evidence()
  corpus_df = build_corpus()

  assert claims_df["claim"].notna().all(), 'null claim'
  assert claims_df["id"].notna().all(), "null claim id"
  assert evidence_df["id"].notna().all(), 'null evidence id'
  assert evidence_df["evidence_wiki_url"].notna().all(), "null evidence url"
  assert corpus_df["id"].notna().all(), "null corpus id"
  assert corpus_df["text"].notna().all(), 'null corpus text'

  claims_df.to_csv(path / "fever_claims_final.csv", index = False)
  claims_df.to_parquet(path / "fever_claims_parquet", index = False)

  evidence_df.to_csv(path / "fever_evidence_final.csv", index = False)
  evidence_df.to_parquet(path / "fever_evidence_parquet", index = False)

  corpus_df.to_csv(path / "fever_corpus_final.csv", index = False)
  corpus_df.to_parquet(path / "fever_corpus_parquet", index = False)

  build_gold_map(evidence_df, json_out=str(path / "fever_gold.json"))


if __name__ == "__main__":
  main()