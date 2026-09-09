# Pipeline de Tokenização Streaming

Pipeline para leitura, tokenização e armazenamento de grandes datasets de tradução utilizando:

- TSV como fonte dos textos;
- Parquet como fonte dos splits (`train`, `val`, `test`);
- Tokenizers para tokenização;
- multiprocessing para paralelização;
- NPZ para armazenamento dos tokens;
- Parquet para armazenamento dos metadados;
- checkpoint para retomada automática;
- arquivos temporários para garantir escrita segura;
- processamento em streaming para evitar consumo excessivo de RAM.

---

## 1. Objetivo

O objetivo deste script é tokenizar um dataset grande sem precisar carregar todos os índices ou textos em memória.

A arquitetura evita a criação de estruturas como:

```python
split_indices = {
    "train": {
        ...
    },
    "val": {
        ...
    },
    "test": {
        ...
    }
}
```

que podem consumir uma quantidade significativa de memória quando o dataset possui milhões de exemplos.

Em vez disso, o processamento ocorre em batches:

```text
Parquet
   │
   ▼
batch de índices
   │
   ▼
TSV correspondente
   │
   ▼
batch de textos
   │
   ▼
workers
   │
   ▼
tokenização
   │
   ▼
NPZ + metadata
```

Somente uma pequena quantidade dos dados fica em memória por vez.

---

# 2. Arquivos de entrada

O pipeline utiliza dois arquivos principais.

## 2.1 TSV

O TSV contém os textos que serão tokenizados.

Exemplo conceitual:

```text
dataset_id<SEP>index<METADATA>texto_origem<SEP>texto_destino
```

Exemplo:

```text
dataset_01_en_pt<SEP>123<METADATA>Hello world<SEP>Olá mundo
```

O script extrai:

```python
dataset_id = "dataset_01_en_pt"
index = 123
text1 = "Hello world"
text2 = "Olá mundo"
```

---

## 2.2 Parquet de splits

O Parquet contém a associação entre cada exemplo e seu split.

As colunas esperadas são:

```text
dataset_id
index
split
```

Exemplo:

| dataset_id | index | split |
|---|---:|---|
| dataset_01_en_pt | 0 | train |
| dataset_01_en_pt | 1 | train |
| dataset_01_en_pt | 2 | val |
| dataset_01_en_pt | 3 | test |

---

# 3. Importante: ordem do TSV e Parquet

A versão atual do pipeline trabalha assumindo que o TSV e o Parquet possuem os exemplos na mesma ordem.

Por exemplo:

### TSV

```text
linha 0 → dataset_A / 0
linha 1 → dataset_A / 1
linha 2 → dataset_A / 2
```

### Parquet

```text
linha 0 → dataset_A / 0
linha 1 → dataset_A / 1
linha 2 → dataset_A / 2
```

O script verifica essa correspondência.

Se encontrar algo como:

```text
TSV:
dataset_A / 10

Parquet:
dataset_A / 15
```

o processamento é interrompido.

Isso é proposital.

É melhor interromper o processamento do que associar silenciosamente o `split` de um exemplo ao texto de outro.

---

# 4. Configuração

As principais configurações ficam no início do arquivo Python.

```python
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 256

NPZ_CHUNK_SIZE = 100_000

PARQUET_BATCH_SIZE = 10_000

MAX_PENDING = 8

MAX_WORKERS = 8
```

---

## 4.1 `MAX_SRC_LENGTH`

Tamanho máximo da sequência de origem.

```python
MAX_SRC_LENGTH = 256
```

Sequências maiores são truncadas.

Sequências menores recebem padding.

---

## 4.2 `MAX_TGT_LENGTH`

Tamanho máximo da sequência de destino.

```python
MAX_TGT_LENGTH = 256
```

Funciona da mesma maneira que `MAX_SRC_LENGTH`.

---

## 4.3 `NPZ_CHUNK_SIZE`

Quantidade de exemplos armazenados em cada arquivo NPZ.

```python
NPZ_CHUNK_SIZE = 100_000
```

Isso produz arquivos aproximadamente assim:

```text
train_00000.npz
train_00001.npz
train_00002.npz

val_00000.npz
val_00001.npz

test_00000.npz
test_00001.npz
```

Um valor maior significa:

- menos arquivos;
- menos operações de I/O;
- maior consumo de RAM durante o `flush`.

Um valor menor significa:

- menor consumo de RAM;
- mais arquivos;
- mais operações de I/O.

---

# 5. `PARQUET_BATCH_SIZE`

Define quantas linhas são lidas do Parquet por vez.

```python
PARQUET_BATCH_SIZE = 10_000
```

Por exemplo, com:

```text
10.000.000 rows
```

teremos aproximadamente:

```text
10.000.000 / 10.000
= 1.000 batches
```

A fórmula é:

```python
import math

num_batches = math.ceil(
    parquet_rows / PARQUET_BATCH_SIZE
)
```

---

# 6. `MAX_WORKERS`

Define quantos processos serão utilizados para a tokenização.

```python
MAX_WORKERS = 8
```

Por exemplo:

```text
8 workers
     │
     ├── batch 1
     ├── batch 2
     ├── batch 3
     ├── batch 4
     ├── batch 5
     ├── batch 6
     ├── batch 7
     └── batch 8
```

Quando um worker termina, recebe outro batch.

---

# 7. `MAX_PENDING`

Define quantos batches podem ficar simultaneamente aguardando processamento.

```python
MAX_PENDING = 8
```

Normalmente é interessante manter:

```python
MAX_PENDING = MAX_WORKERS
```

ou um valor pequeno acima disso.

Evite colocar valores extremamente altos.

Por exemplo:

```python
MAX_WORKERS = 8
MAX_PENDING = 1000
```

pode fazer muitos batches ficarem acumulados na memória.

---

# 8. Uso de memória

Uma das principais características deste pipeline é evitar estruturas gigantes em RAM.

O script **não cria um `split_indices` contendo todos os índices do dataset**.

Em vez disso, trabalha aproximadamente assim:

```text
10.000 rows
    ↓
batch
    ↓
workers
    ↓
tokens
    ↓
NPZ
    ↓
libera memória
```

O consumo de memória passa a depender principalmente de:

- `PARQUET_BATCH_SIZE`;
- `MAX_PENDING`;
- `MAX_WORKERS`;
- `NPZ_CHUNK_SIZE`;
- tamanho médio dos textos;
- tamanho máximo das sequências.

---

# 9. Tokenização

Cada worker carrega uma instância do tokenizer:

```python
def init_worker(tokenizer_path):

    global _WORKER_TOKENIZER

    _WORKER_TOKENIZER = Tokenizer.from_file(
        tokenizer_path
    )
```

O tokenizer é carregado uma vez por processo.

Isso evita recarregar o tokenizer para cada batch.

---

# 10. Tags de direção

O pipeline adiciona tags dependendo do `dataset_id`.

Para datasets terminando em:

```text
_en_pt
```

é adicionado:

```text
<2pt>
```

ao texto de origem.

Para:

```text
_en_es
```

é adicionado:

```text
<2es>
```

Exemplo:

```text
Hello world
```

vira:

```text
<2pt> Hello world
```

antes da tokenização.

---

# 11. Padding e truncamento

Depois da tokenização, as sequências são ajustadas para o tamanho máximo.

Por exemplo:

```python
MAX_SRC_LENGTH = 256
```

Uma sequência com 300 tokens será truncada:

```text
300 tokens
    ↓
256 tokens
```

Uma sequência com 100 tokens será preenchida:

```text
100 tokens
    ↓
100 tokens + 156 PAD
```

O ID de `<PAD>` é obtido diretamente do tokenizer:

```python
pad_id = tokenizer.token_to_id(
    "<PAD>"
)
```

Se o tokenizer não possuir `<PAD>`, o programa interrompe.

---

# 12. Arquivos NPZ

Os tokens são armazenados em arquivos `.npz`.

Cada arquivo contém:

```python
src_tokens
tgt_tokens
src_lengths
tgt_lengths
```

Exemplo:

```python
data = np.load(
    "train_00000.npz"
)

src_tokens = data["src_tokens"]
tgt_tokens = data["tgt_tokens"]
src_lengths = data["src_lengths"]
tgt_lengths = data["tgt_lengths"]
```

Os arrays são armazenados como:

```python
dtype=np.int32
```

para reduzir o consumo de armazenamento e memória.

---

# 13. Metadados

Para cada NPZ é criado um Parquet correspondente.

Exemplo:

```text
train_00000.npz
train_00000_meta.parquet
```

Os metadados possuem:

```text
dataset_id
index
split
npz_path
npz_index
```

Exemplo:

| dataset_id | index | split | npz_path | npz_index |
|---|---:|---|---|---:|
| dataset_A | 100 | train | train_00000.npz | 0 |
| dataset_A | 101 | train | train_00000.npz | 1 |
| dataset_A | 102 | train | train_00000.npz | 2 |

Assim é possível localizar exatamente onde determinado exemplo foi armazenado.

---

# 14. Estrutura dos arquivos

Ao final, o diretório será aproximadamente:

```text
tokenized/
│
├── train_00000.npz
├── train_00001.npz
├── train_00002.npz
│
├── val_00000.npz
├── val_00001.npz
│
├── test_00000.npz
├── test_00001.npz
│
├── checkpoint.json
├── processing_config.json
│
├── analise_textos_tokenized_split.parquet
│
└── metadata_parts/
    ├── train_00000_meta.parquet
    ├── train_00001_meta.parquet
    ├── train_00002_meta.parquet
    ├── val_00000_meta.parquet
    ├── val_00001_meta.parquet
    ├── test_00000_meta.parquet
    └── ...
```

---

# 15. Retomada automática

O pipeline possui checkpoint.

O arquivo:

```text
checkpoint.json
```

contém informações como:

```json
{
  "tsv_rows_consumed": 5000000,
  "processed_examples": 4998123,
  "train_chunks": 42,
  "val_chunks": 7,
  "test_chunks": 3
}
```

Isso permite saber onde o processamento estava.

Ao iniciar novamente, o programa lê:

```python
checkpoint = load_checkpoint()
```

e recupera:

```python
start_row = checkpoint[
    "tsv_rows_consumed"
]
```

Assim, o TSV não precisa ser processado novamente desde o início.

---

# 16. Arquivos temporários

O programa não grava diretamente no arquivo final.

Por exemplo, para:

```text
train_00042.npz
```

primeiro é criado:

```text
train_00042.npz.tmp
```

Depois que a gravação termina corretamente:

```text
train_00042.npz.tmp
        ↓
train_00042.npz
```

O mesmo mecanismo é utilizado para os metadados.

Isso evita considerar um arquivo parcialmente escrito como um chunk válido.

---

# 17. Escrita atômica

O método utilizado é:

```python
os.replace(
    temp_path,
    final_path
)
```

A ideia é que o arquivo final só apareça depois que o arquivo temporário estiver completo.

Portanto, se o processo morrer durante:

```text
train_00042.npz.tmp
```

o próximo processamento pode detectar que o arquivo final:

```text
train_00042.npz
```

não existe e tratar aquele chunk como incompleto.

---

# 18. Checkpoint seguro

O checkpoint também é escrito de maneira atômica.

Em vez de:

```text
checkpoint.json
```

ser sobrescrito diretamente, o programa cria:

```text
checkpoint.json.tmp
```

e depois executa:

```python
os.replace(
    temp_path,
    checkpoint_path
)
```

Isso reduz o risco de ficar com um JSON corrompido caso o processo seja interrompido durante a escrita.

---

# 19. O que acontece se o processo for interrompido?

Suponha que o processamento esteja assim:

```text
train_00000.npz ✓
train_00001.npz ✓
train_00002.npz ✓
train_00003.npz ✓
train_00004.npz → processando
```

Se a máquina desligar:

```text
train_00000.npz ✓
train_00001.npz ✓
train_00002.npz ✓
train_00003.npz ✓
train_00004.npz.tmp
```

Ao reiniciar:

1. O checkpoint é carregado.
2. Os chunks completos permanecem.
3. Arquivos `.tmp` são removidos.
4. O processamento continua a partir do checkpoint.
5. O chunk incompleto é refeito.

---

# 20. Verificação dos chunks

Antes de consolidar os metadados, o script verifica se todos os metadata possuem seu respectivo NPZ.

Por exemplo:

```text
train_00042_meta.parquet
```

deve possuir:

```text
train_00042.npz
```

Se estiver faltando, o programa acusa o problema.

Isso evita consolidar um dataset incompleto.

---

# 21. Consolidação dos metadados

Durante o processamento, os metadados são mantidos separados:

```text
metadata_parts/
    train_00000_meta.parquet
    train_00001_meta.parquet
    train_00002_meta.parquet
    ...
```

No final eles são consolidados em:

```text
analise_textos_tokenized_split.parquet
```

A consolidação também é feita de forma incremental.

O código não precisa carregar todos os Parquets simultaneamente em memória.

---

# 22. Barra de progresso

Para mostrar progresso baseado em **rows**, utilize:

```python
pbar = tqdm(
    total=parquet_rows,
    desc="Processando",
    unit="row",
)
```

E, após processar cada DataFrame:

```python
pbar.update(
    len(parquet_df)
)
```

Assim, se o Parquet possuir:

```text
10.000.000 rows
```

a barra mostrará algo como:

```text
Processando: 35% | 3.500.000/10.000.000
```

---

# 23. Quantidade de batches

A quantidade de batches não depende de `MAX_WORKERS`.

Ela depende de:

```python
PARQUET_BATCH_SIZE
```

A fórmula é:

```python
import math

num_batches = math.ceil(
    parquet_rows /
    PARQUET_BATCH_SIZE
)
```

Por exemplo:

```text
parquet_rows = 10.000.000
PARQUET_BATCH_SIZE = 10.000
```

resulta em:

```text
1.000 batches
```

Se:

```text
MAX_WORKERS = 8
```

isso significa apenas que até 8 batches podem ser processados simultaneamente.

```text
1.000 batches
       │
       ▼
┌───────────────────────────────┐
│  Worker 1 → batch             │
│  Worker 2 → batch             │
│  Worker 3 → batch             │
│  Worker 4 → batch             │
│  Worker 5 → batch             │
│  Worker 6 → batch             │
│  Worker 7 → batch             │
│  Worker 8 → batch             │
└───────────────────────────────┘
```

Quando um worker termina, recebe o próximo batch.

---

# 24. Ajustando performance

Os principais parâmetros para ajustar são:

```python
PARQUET_BATCH_SIZE
NPZ_CHUNK_SIZE
MAX_WORKERS
MAX_PENDING
```

Uma configuração inicial razoável seria:

```python
PARQUET_BATCH_SIZE = 10_000
NPZ_CHUNK_SIZE = 100_000
MAX_WORKERS = 8
MAX_PENDING = 8
```

Se houver muita RAM disponível, pode-se aumentar:

```python
PARQUET_BATCH_SIZE
```

Se a CPU estiver subutilizada, pode-se aumentar:

```python
MAX_WORKERS
```

Se houver pressão de memória, reduza:

```python
PARQUET_BATCH_SIZE
MAX_PENDING
NPZ_CHUNK_SIZE
```

---

# 25. Cuidados com memória

Não aumente todos os parâmetros simultaneamente.

Por exemplo, esta configuração pode consumir bastante RAM:

```python
PARQUET_BATCH_SIZE = 1_000_000
NPZ_CHUNK_SIZE = 1_000_000
MAX_WORKERS = 32
MAX_PENDING = 32
```

Cada worker pode possuir uma quantidade considerável de textos/tokenizações em memória.

Para datasets muito grandes, é preferível começar conservadoramente:

```python
PARQUET_BATCH_SIZE = 10_000
NPZ_CHUNK_SIZE = 100_000
MAX_WORKERS = 8
MAX_PENDING = 8
```

e aumentar gradualmente.

---

# 26. Dependências

O script utiliza:

```text
numpy
pandas
pyarrow
tqdm
tokenizers
```

Instalação:

```bash
pip install numpy pandas pyarrow tqdm tokenizers
```

---

# 27. Execução

Execute normalmente:

```bash
python tokenize_dataset.py
```

O script irá:

```text
1. verificar diretórios
2. verificar configuração
3. carregar tokenizer
4. verificar o Parquet
5. carregar checkpoint
6. retomar o processamento
7. ler TSV + Parquet em streaming
8. enviar batches para os workers
9. tokenizar
10. salvar NPZ
11. salvar metadata
12. atualizar checkpoint
13. verificar os chunks
14. consolidar metadata
```

---

# 28. Começar novamente do zero

Se quiser descartar completamente o processamento anterior, remova:

```text
checkpoint.json
processing_config.json
```

e também os arquivos gerados:

```text
*.npz
metadata_parts/*.parquet
analise_textos_tokenized_split.parquet
```

Uma forma simples seria remover todo o diretório:

```bash
rm -rf /caminho/para/tokenized
```

Depois execute o script novamente.

**Cuidado:** isso apaga os dados tokenizados existentes.

---

# 29. Resumo da arquitetura

O pipeline foi projetado para evitar o seguinte:

```text
TSV
 ↓
carrega tudo
 ↓
cria split_indices gigantes
 ↓
tokeniza
```

Em vez disso:

```text
                 TSV
                  │
             streaming
                  │
                  ▼
              pequenos
               batches
                  │
                  │
Parquet ──────────┤
                  │
                  ▼
             multiprocessing
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
     Worker    Worker    Worker
        │         │         │
        └─────────┼─────────┘
                  ▼
              Tokenização
                  │
                  ▼
             NPZ chunks
                  │
                  ▼
          Metadata Parquet
                  │
                  ▼
             Checkpoint
```

O resultado é um pipeline:

- **streaming**;
- **paralelizado**;
- **incremental**;
- **retomável**;
- **resistente a interrupções**;
- sem `split_indices` gigante em RAM;
- com armazenamento em chunks;
- com validação de consistência entre TSV e Parquet.
