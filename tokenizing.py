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

for name, dataset in datasets.items():
    for vocab_size in [10000, 50000]:
        tokenizer: Tokenizer = Tokenizer.from_file(f'artifacts/tokenizer_{vocab_size}.json')
        print(f'Tokenizing {name}')
        
        tqdm.pandas(desc=f'Tokenizing EN')
        dataset[f'en_tokens_{vocab_size}'] = dataset.en.progress_apply(lambda x: tokenizer.encode(x).ids)
        tqdm.pandas(desc=f'Tokenizing PT')
        dataset[f'pt_tokens_{vocab_size}'] = dataset.pt.progress_apply(lambda x: tokenizer.encode(x).ids)
        
        print('Exemplo: ', dataset[f'en_tokens_{vocab_size}'].iloc[0], '->', dataset[f'pt_tokens_{vocab_size}'].iloc[0])
    dataset.to_parquet(f'data/tokenized_{name}.parquet', index=False)