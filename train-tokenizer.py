import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


print('Loading data...')
df_train = pd.read_csv('data/train.csv')

print('Declaring tokenizer...')
for vocab_size in [500, 1000, 5000, 10000, 25000, 30000, 50000]:
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<BOS>", "<EOS>"],
    )

    print('Training tokenizer...')
    tokenizer.train_from_iterator(
        df_train['pt'].tolist() + df_train['en'].tolist(),
        trainer=trainer
    )

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <SEP> $B:1 <EOS>:1",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<SEP>", tokenizer.token_to_id("<SEP>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    print('Saving tokenizer...')
    tokenizer.save(f'artifacts/tokenizer_{vocab_size}.json')