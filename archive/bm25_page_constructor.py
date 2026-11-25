from pathlib import Path
import pandas as pd
import json
import unicodedata, re

PROC = Path(r"C:\Users\mukun\crisis-claim-analysis\data\processed\fever_corpus_parquet")
OUTDIR = Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\bm25_sentences")
JSONL = OUTDIR / "jsonl" / "sentences.jsonl"
OFFMAP = OUTDIR / "page_offsets.json"

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return re.sub(r"\s+", "_", s)  

def main():
  (OUTDIR / "jsonl").mkdir(parents=True, exist_ok=True)
  df = pd.read_parquet(PROC)[["id", "line", "text"]].dropna()
  df["id"] = df["id"].map(norm_title)
  df["line"] = df["line"].astype(int)
  df["text"] = df["text"].astype(str)
  
  try:
    with OFFMAP.open("r", encoding="utf-8") as f:
      page_offset = json.load(f)
    
  except FileNotFoundError:
    page_offset = {}

  with JSONL.open("w", encoding="utf-8") as f:
    for row in df.itertuples(index=False):
      #we also check now to see if it is line-offset or not
      offset = int(page_offset.get(row.id, 0))
      line_aligned = row.line + offset
      docid    = f"{row.id}#{line_aligned}"
      contents = f"{row.id.replace('_',' ')} [SEP] {row.id} [SEP] {row.text}"
      f.write(json.dumps({"id" : docid, "contents" : contents}, ensure_ascii=False) + "\n")
  print("sentences:", len(df))

if __name__ == "__main__":
  main()
