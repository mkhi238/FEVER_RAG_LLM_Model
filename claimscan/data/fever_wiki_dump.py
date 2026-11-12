from datasets import load_dataset
import pandas as pd
from common import normalize_text

def build_corpus():
    wiki_dataset = load_dataset("fever", "wiki_pages", split="wikipedia_pages")
    df = pd.DataFrame(wiki_dataset)
    print(df.columns)
    rows = []

    for _, row in df.iterrows():
        title = row['id']
        if pd.isna(row['lines']):
            continue
        try:
          for i in row['lines'].split('\n'):
              if not i.strip():
                  continue
              num, line = i.split("\t", 1)
              line = line.strip()
              line = normalize_text(line)
              if line:
                  rows.append((title, int(num), line))
        except ValueError:
            continue
    
    corpus = pd.DataFrame(rows, columns = ["id","line","text"]).drop_duplicates(subset = ['id', 'line']).reset_index(drop=True)

    return corpus

#id is the wiki page name/title
#line is the sentence index within the page
#text is the sentence context
