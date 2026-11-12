from datasets import load_dataset
import pandas as pd
from common import clean_data


canon = {'false statement': 'FALSE', 
         'true statement': 'TRUE',}

def main():

    dataset = load_dataset("UKPLab/liar", trust_remote_code=True)
    train = clean_data(dataset['train'], 'train', canon, text_col='text', label_col='label_text')
    validation = clean_data(dataset['validation'], 'validation', canon, text_col='text', label_col='label_text')
    test = clean_data(dataset['test'], 'test', canon, text_col='text', label_col='label_text')
    df = pd.concat([train, validation, test], ignore_index=True)
    priority = {'train': 0, 'validation': 1, 'test': 2}
    df['prio'] = df['split'].map(priority)
    df = df.sort_values(['claim', 'prio'])
    df = df.drop_duplicates(subset=["claim"], keep="first")
    df = df.drop('prio', axis = 1).reset_index(drop = True)
    df.rename(columns={'label':'label_text'}, inplace=True)
    df = df[['claim', 'label_text', 'labels']]
    df.to_csv("liar_cleaned.csv", index=False)



if __name__ == "__main__":
    main()
