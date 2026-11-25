from pathlib import Path
import pandas as pd
import json


PROC = Path(r"C:\Users\mukun\crisis-claim-analysis\data\processed\fever_corpus_parquet")
OUTDIR = Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\bm25_sentences")
JSONL = OUTDIR / "jsonl" / "sentences.jsonl"

def main():
  (OUTDIR / "jsonl").mkdir(parents=True, exist_ok=True)
  df = pd.read_parquet(PROC)
  df["id"] = df["id"].astype(str)
  df["line"] = df["line"].astype(int)
  df["text"] = df["text"].astype(str)
  with JSONL.open("w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
      docid = f"{row['id']}#{row['line']}"
      f.write(json.dumps({"id" : docid, "contents" : str(row["text"])}, ensure_ascii=False) + "\n")
  print("pages:", len(df))

if __name__ == "__main__":
  main()