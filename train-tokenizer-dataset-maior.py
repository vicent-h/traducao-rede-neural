import pandas as pd
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import os
from tqdm import tqdm
import math
import gc
import numpy as np

# ============================================================
# CONFIGURAÇÕES
# ============================================================
TSV_PATH = '/media/alvarinho/dados/Datasets/refined/traducao/analise_textos.tsv'
SPLIT_PARQUET_PATH = '/media/alvarinho/dados/Datasets/refined/traducao/analise_textos_split.parquet'

# ============================================================
# FUNÇÕES DE LEITURA E PREPARAÇÃO
# ============================================================
def load_train_indices(parquet_path, batch_size=1_000_000, target_sample_size=1_000_000):
    print('Carregando metadados de split em lotes...')
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Dicionário temporário para guardar arrays numéricos, que são muito leves
    temp_indices = {}
    split_counts = {}
    
    total_linhas = parquet_file.metadata.num_rows
    
    # Lemos APENAS as 3 colunas necessárias para o sorteio, poupando muita RAM e CPU
    with tqdm(total=total_linhas, desc='Mapeando indices', unit='linhas') as pbar:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, 
            columns=['dataset_id', 'index', 'split']
        ):
            linhas_neste_lote = batch.num_rows 
            df_chunk = batch.to_pandas()
            
            # Contagem global
            counts = df_chunk['split'].value_counts().to_dict()
            for split_name, count in counts.items():
                split_counts[split_name] = split_counts.get(split_name, 0) + count
                
            # Filtra apenas o split de treino
            df_train = df_chunk[df_chunk['split'] == 'train']
            
            # Agrupa e guarda os arrays do numpy
            for dataset_id, group in df_train.groupby('dataset_id', observed=True):
                if dataset_id not in temp_indices:
                    temp_indices[dataset_id] = []
                
                temp_indices[dataset_id].append(group['index'].values)
                
            del df_chunk
            del df_train
            gc.collect()
            
            pbar.update(linhas_neste_lote)
            
    print('\nDistribuição total encontrada nos dados:')
    for split_name, count in split_counts.items():
        print(f"  - {split_name}: {count:,}".replace(',', '.'))

    # ============================================================
    # AMOSTRAGEM ESTRATIFICADA
    # ============================================================
    print(f'\nRealizando amostragem estratificada para ~{target_sample_size:,} textos...'.replace(',', '.'))
    
    train_indices = {}
    total_train_available = split_counts.get('train', 0)
    
    # Se a base de treino for menor que o alvo, usamos tudo. Se não, calculamos a fração.
    if total_train_available <= target_sample_size:
        sampling_fraction = 1.0
    else:
        sampling_fraction = target_sample_size / total_train_available

    total_amostrado = 0
    
    for dataset_id, arrays_list in temp_indices.items():
        # Une todos os mini-arrays desse dataset que vieram dos lotes
        all_indices = np.concatenate(arrays_list)
        
        # Calcula a cota deste dataset preservando a proporção exata
        k = math.ceil(len(all_indices) * sampling_fraction)
        
        if k > 0:
            # np.random.choice faz o sorteio sem reposição
            sampled = np.random.choice(all_indices, size=k, replace=False)
            
            # Agora sim, convertemos apenas a amostra reduzida para set (para a busca rápida no TSV)
            train_indices[dataset_id] = set(sampled)
            total_amostrado += len(sampled)
            
            print(f"  - {dataset_id}: selecionou {len(sampled):,} de {len(all_indices):,}".replace(',', '.'))
        else:
            train_indices[dataset_id] = set()
            
        # Limpa o array gigante original da RAM
        del all_indices

    gc.collect()
    print(f"\nTotal final amostrado para o treinamento: {total_amostrado:,}".replace(',', '.'))
    
    return train_indices

def get_text_iterator(tsv_path, train_indices, batch_size=10000):
    """
    Lê o arquivo TSV linha por linha, faz o split pelas marcações,
    injeta a tag de tradução no texto de origem e yield dos textos.
    """
    batch = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        next(f) # Pula o cabeçalho original
        
        for line in f:
            line = line.rstrip('\n')
            
            try:
                meta_part, texts_part = line.split('<METADATA>')
                dataset_id, idx_str = meta_part.split('<SEP>')
                text1, text2 = texts_part.split('<SEP>')
                
                # Só inclui o par de textos se ele foi sorteado na nossa amostragem
                dataset_set = train_indices.get(dataset_id)
                if dataset_set is not None and int(idx_str) in dataset_set:
                    
                    # ====================================================
                    # INJEÇÃO DA TAG DE TRADUÇÃO (Approach 1)
                    # ====================================================
                    # Verifica o sufixo do dataset_id para saber a direção[cite: 2]
                    if dataset_id.endswith('_en_pt'):
                        text1_modificado = f"<2pt> {text1}"
                    elif dataset_id.endswith('_en_es'):
                        text1_modificado = f"<2es> {text1}"
                    else:
                        text1_modificado = text1 # Fallback de segurança
                        
                    # Adiciona o texto em inglês (com a tag) e o texto alvo ao lote
                    batch.extend([text1_modificado, text2])
                    
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
                    
            except ValueError:
                # Passa silenciosamente por linhas mal formatadas
                continue
                
    if batch:
        yield batch


# ============================================================
# MAIN - TREINAMENTO DO TOKENIZADOR
# ============================================================
if __name__ == '__main__':
    # 1. Carrega o mapa de treino e printa as quantidades por split
    train_indices = load_train_indices(SPLIT_PARQUET_PATH)

    print('\nDeclarando tokenizer...')
    
    # 2. O loop mantém-se igual, iterando sobre os tamanhos de vocabulário
    for vocab_size in [30_000, 50_000, 60_000, 120_000, 150_000, 180_000, 200_000]:
        print(f'\n{"="*50}')
        print(f'--- Treinando para vocab_size: {vocab_size} ---')
        print(f'{"="*50}')
        
        tokenizer = Tokenizer(
            models.BPE(
                unk_token="<UNK>",
                end_of_word_suffix="</w>"
            )
        )

        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>", "<BOS>", "<EOS>"],
            end_of_word_suffix="</w>",
        )

        print('Iniciando o treinamento do tokenizador (isso pode levar algum tempo)...')
        
        # 3. Passamos o iterador que lê o TSV sob demanda e filtra pelo split
        tokenizer.train_from_iterator(
            get_text_iterator(TSV_PATH, train_indices), 
            trainer=trainer,
        )

        # 4. Adicionando o post-processor
        tokenizer.post_processor = processors.TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <SEP> $B:1 <EOS>:1",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<SEP>", tokenizer.token_to_id("<SEP>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )

        # 5. Salva os artefatos
        os.makedirs('artifacts', exist_ok=True)
        print(f'Salvando tokenizer_{vocab_size}...')
        tokenizer.save(f'artifacts/tokenizer_en_pt_es_{vocab_size}.json')
        
    print('\nTodos os tokenizadores foram treinados e salvos com sucesso!')