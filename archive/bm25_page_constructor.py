from pathlib import Path
import pandas as pd
import json
import unicodedata, re

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
OFFMAP = OUTDIR / "page_offsets.json"

def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    return re.sub(r"\s+", "_", s)  

def main():
  (OUTDIR / "jsonl").mkdir(parents=True, exist_ok=True)
  df = pd.read_parquet(PROC)[["id", "line", "text"]].dropna()
  #apply(lambda x: "\n".join(map(str, x))) is sentence by sentence string conversion + join
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
      #before, each Lucene “document” was a whole page and its id was just the page title; now, each Lucene “document” is a single sentence and its id is "{page}#{line}" so you can map hits directly to FEVER’s (page_id, line) gold pairs.
      #we also check now to see if it is line-offset or not
      offset = int(page_offset.get(row.id, 0))
      line_aligned = row.line + offset
      docid    = f"{row.id}#{line_aligned}"
      contents = f"{row.id.replace('_',' ')} [SEP] {row.id} [SEP] {row.text}"
      f.write(json.dumps({"id" : docid, "contents" : contents}, ensure_ascii=False) + "\n")
  print("sentences:", len(df))

if __name__ == "__main__":
  main()

#This script does the data preperation
#It loads your processed FEVER corpus parquet (one row = one sentence).
#It normalizes the page titles (e.g. "Barack Obama" → "Barack_Obama").
#It assigns a document ID per sentence: "{page_title}#{line_number}" - Page = "Obama", line = 0, text = "Barack Obama was born..." becomes
#{"id": "Obama#0", "contents": "Obama [SEP] Obama [SEP] Barack Obama was born..."}
#writes all sentences to a jsonl file
#This script does NOT build a searchable index. It just dumps the raw text into a format that Pyserini can ingest.

#The indexing command (actually wehre the lucene index is built)
#Reads the sentences.jsonl you just wrote.
#Each line is treated as one Lucene “document”.
#The id field becomes docid.
#The contents field is what Lucene tokenizes and indexes.
#Builds an inverted index (mapping tokens → list of docids + positions).

#JSONL builder script → creates the raw input in the right format.
#Indexing command → actually builds the optimized search index that LuceneSearcher can use at query time
