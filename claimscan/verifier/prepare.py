import json
import random
import unicodedata
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import tempfile
import shutil
import os

#Creates model training data 

from claimscan.retrieval.faiss_retriever import FAISSRetriever
from claimscan.config import config

def norm_title(s):

    s = unicodedata.normalize("NFKC", str(s)).strip()
    return re.sub(r"\s+", "_", s)

def load_fever_data():
    df_claims = pd.read_parquet(config.claims_file)
    df_corpus = pd.read_parquet(config.corpus)
    df_evidence = pd.read_parquet(config.evidence)

    return df_claims, df_corpus, df_evidence

def get_evidence_text(gold_evidence, corpus_df):
    sentences = []
    for _, row in gold_evidence.iterrows():
        wiki_url = norm_title(row['evidence_wiki_url'])
        sentence_id = row['evidence_sentence_id']

        #find gold evidence in corpus 
        match = corpus_df[
            (corpus_df['id'] == wiki_url) & (corpus_df['line']==sentence_id)
        ]

        if not match.empty:
            sentences.append(match.iloc[0]['text'])
    return sentences

def get_negative_text(claim_text, gold_evidence, corpus_df, retriever):
    try:
        results = retriever.search(claim_text, k = 20)

        gold_pairs = set()
        for _, row in gold_evidence.iterrows():
            wiki_url = norm_title(row['evidence_wiki_url'])
            sentence_id = row['evidence_sentence_id']
            gold_pairs.add((wiki_url, sentence_id))
        
        negatives = []
        for result in results:
            page = norm_title(result['id'])
            line = int(result['line'])

            if (page, line) in gold_pairs:
                continue

            match = corpus_df[
                (corpus_df['id'] == page) & (corpus_df['line'] == line)
            ]

            if not match.empty:
                negatives.append(match.iloc[0]['text'])
                if len(negatives) >= config.negative_samples_per_claim:
                    break

        return negatives
    except Exception as e:
        print(f"Error {e}")
        return []
    

def create_training_data(claims_df, evidence_df, corpus_df, retriver, target_amount_per_class = 2500):
    train_claims = claims_df[claims_df['split'] == "train"].sample(n=75000, random_state=19)

    true_examples = []
    false_examples = []
    nei_examples = []

    for _, row in tqdm(train_claims.iterrows(), total=len(train_claims), desc="Processing claims"):
        claim_id = int(row['id'])
        claim_text = str(row['claim'])
        label = str(row['label_text'])

        gold_evidence = evidence_df[evidence_df['id'] == claim_id]

        #if there is evidence and evidence is true, collect true evidence 
        if not gold_evidence.empty and label == 'TRUE' and len(true_examples) < target_amount_per_class:
            sentences = get_evidence_text(gold_evidence, corpus_df)
            if sentences:
                true_examples.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "evidence": sentences[:config.max_evidence_sentences],
                    "label" : label
                })

        if not gold_evidence.empty and label == 'FALSE' and len(false_examples) < target_amount_per_class:
            sentences = get_evidence_text(gold_evidence, corpus_df)
            if sentences:
                false_examples.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "evidence": sentences[:config.max_evidence_sentences],
                    "label" : label
                })

        if len(nei_examples) < target_amount_per_class:
            neg_sentences = get_negative_text(claim_text, gold_evidence, corpus_df, retriver)
            if neg_sentences:
                nei_examples.append({
                    "claim_id": claim_id,
                    "claim": claim_text,
                    "evidence": neg_sentences[:config.max_evidence_sentences],
                    "label" : "NOT ENOUGH INFO"
                })

        if len(true_examples) >= target_amount_per_class and len(false_examples) >= target_amount_per_class and len(nei_examples) >= target_amount_per_class:
            break
    
    print(f"\nCollected: TRUE={len(true_examples)}, FALSE={len(false_examples)}, NEI={len(nei_examples)}")

    examples = true_examples + false_examples + nei_examples
    return examples

def main():
    claims_df, corpus_df, evidence_df = load_fever_data()

    retriver  = FAISSRetriever(config.retriever_index_dir, config.retriever_model)

    examples = create_training_data(claims_df, evidence_df, corpus_df, retriver )
    
    try:
        backup_path = Path("training_data.json")
        with open(backup_path, "w") as f:
            json.dump(examples, f, indent=2)
        print(f"BACKUP SUCCESS: Saved to {backup_path.absolute()}")
        return
        
    except Exception as e:
        print(f"Failed as {e}")
    

if __name__ == "__main__":
    main()

#Output:
'''
JSON:
  {
    "claim_id": 75397,
    "claim": "Barack Obama was born in Hawaii",
    "evidence": ["Barack Obama was born in Honolulu, Hawaii, on August 4, 1961."],
    "label": "TRUE"
  },
  {
    "claim_id": 75397,
    "claim": "Barack Obama was born in Hawaii", 
    "evidence": ["Barack Obama graduated from Harvard Law School."],
    "label": "NOT ENOUGH INFO"
  },
  {
    "claim_id": 185043,
    "claim": "Danny Brown is an American rapper",
    "evidence": ["Daniel Dewan Sewell, better known by his stage name Danny Brown, is an American rapper."],
    "label": "TRUE"
  }
]
'''
