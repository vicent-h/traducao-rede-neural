import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tqdm import tqdm

print('Loading data...')
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
df_eval = pd.read_csv('data/eval.csv')

datasets = {
    'train': df_train, 
    'test': df_test, 
    'eval': df_eval}

tokenizer: Tokenizer = Tokenizer.from_file('artifacts/tokenizer_10000.json')

for name, dataset in datasets.items():
    print(f'Tokenizing {name}')
    
    tqdm.pandas(desc=f'Tokenizing EN')
    dataset['en_tokens'] = dataset.en.progress_apply(lambda x: tokenizer.encode(x).ids)
    tqdm.pandas(desc=f'Tokenizing PT')
    dataset['pt_tokens'] = dataset.pt.progress_apply(lambda x: tokenizer.encode(x).ids)

    dataset.to_csv(f'data/tokenized_{name}.csv', index=False)