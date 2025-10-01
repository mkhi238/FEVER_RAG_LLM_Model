import unicodedata, re
import pandas as pd
import json


def normalize_text(s):
    s = unicodedata.normalize("NFKC", str(s)).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def clean_data(dataset, split_name, canon = None, text_col = "claim", label_col = "label", id_col = "id"):
    df = dataset.to_pandas() if hasattr(dataset, "to_pandas") else dataset.copy()
    rename = {text_col: "claim", label_col: "label"}
    if id_col in df.columns:
        rename[id_col] = "id"
    df = df.rename(columns = rename)
    df['claim'] = df['claim'].apply(normalize_text)
    if canon:
        df['label'] = df['label'].astype(str).str.lower().map(canon)
    df = df.dropna(subset = ['claim', 'label']).copy()
    df['split'] = split_name
    return df

def build_gold_map(df, json_out):
    #evidence_wiki_url is a name (like Barack Obama)
    #evidence_sentence_id is a sentence numebr on a page where the evidence is (-1 if it cant be found, numeric otherwise)
    df = df.dropna(subset = ["id", "evidence_wiki_url", "evidence_sentence_id"])
    df["id"] = df["id"].astype(int)
    df["evidence_sentence_id"] = df["evidence_sentence_id"].astype(int)

    gold_map = {}
    for claim_tag, group in df.groupby("id"):
        pairs = []
        seen = set()
        for _, row in group.iterrows():
            pair = (row["evidence_wiki_url"], row["evidence_sentence_id"])
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        
        gold_map[claim_tag] = pairs

    if json_out:
        gold_str_keys = {}
        for k, v in gold_map.items():
            claim_id = str(k)
            gold_str_keys[claim_id] = v

        
        with open(json_out, "w", encoding='utf-8') as f:
            json.dump(gold_str_keys, f, ensure_ascii=False, indent=2)

