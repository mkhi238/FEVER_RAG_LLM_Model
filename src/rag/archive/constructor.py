from pathlib import Path
import pandas as pd
import json


#id        line   text
#--------  ----   --------------------
#Obama     0      Barack Obama was born...
#Obama     1      He served as senator...
#Cats      0      Cats are small mammals...

#meta is id (or the page identifier)
#{"id": "Obama", "contents": "Barack Obama was born...\nHe served..."}
#{"id": "Cats", "contents": "Cats are small mammals...\nThey are kept..."}

#pages.jsonl (for BM25)
#Format: {"id": "Obama", "contents": "full page text..."}
#Purpose: BM25 indexes page-level text.
#Stores: concatenated text per page.

#output: What it is: A JSONL file with one JSON object per Wikipedia page.
#BM25 searches this index to find the page that looks relevant to this query.
#Why needed: BM25 only works at the page level, so it needs one big text blob per page.

PROC = Path(r"C:\Users\mukun\crisis-claim-analysis\data\processed\fever_corpus_parquet")
OUTDIR = Path(r"C:\Users\mukun\crisis-claim-analysis\artifacts\bm25_sentences")
JSONL = OUTDIR / "jsonl" / "sentences.jsonl"

def main():
  (OUTDIR / "jsonl").mkdir(parents=True, exist_ok=True)
  df = pd.read_parquet(PROC)
  #apply(lambda x: "\n".join(map(str, x))) is sentence by sentence string conversion + join
  df["id"] = df["id"].astype(str)
  df["line"] = df["line"].astype(int)
  df["text"] = df["text"].astype(str)
  with JSONL.open("w", encoding="utf-8") as f:
    for idx, row in df.iterrows():
      #before, each Lucene “document” was a whole page and its id was just the page title; now, each Lucene “document” is a single sentence and its id is "{page}#{line}" so you can map hits directly to FEVER’s (page_id, line) gold pairs.
      docid = f"{row['id']}#{row['line']}"
      f.write(json.dumps({"id" : docid, "contents" : str(row["text"])}, ensure_ascii=False) + "\n")
  print("pages:", len(df))

if __name__ == "__main__":
  main()