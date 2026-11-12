from datasets import load_dataset
import pandas as pd
from common import clean_data

canon = {'supports': 'TRUE', 
         'refutes': 'FALSE',
         'not enough info': 'NOT ENOUGH INFO',
         'nei': 'NOT ENOUGH INFO'}

def build_fever_evidence():
    dataset = load_dataset("fever", "v1.0")
    train = dataset['train'].to_pandas()[['id','evidence_annotation_id','evidence_wiki_url','evidence_sentence_id']]
    validation = dataset['labelled_dev'].to_pandas()[['id','evidence_annotation_id','evidence_wiki_url','evidence_sentence_id']]
    test = dataset['paper_test'].to_pandas()[['id','evidence_annotation_id','evidence_wiki_url','evidence_sentence_id']]
    df = pd.concat([train, validation, test], ignore_index=True)
    return df
    

