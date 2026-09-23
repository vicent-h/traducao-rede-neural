# Processamento de Tokenização com Shards

## 1. Objetivo

Este processamento foi desenvolvido para trabalhar com um dataset grande, em que:

- o **TSV não cabe inteiro na RAM**;
- o **Parquet de metadados cabe na RAM**;
- a tokenização é relativamente pesada e deve ser paralelizada;
- o processamento pode ser interrompido e posteriormente retomado;
- não queremos criar um arquivo separado para cada exemplo;
- queremos evitar perder uma grande quantidade de exemplos já tokenizados quando o processo é interrompido.

O fluxo adotado é:

```text
TSV ────────────────┐
                    ├──> batches ──> workers ──> resultados
Parquet em RAM ─────┘                              │
                                                   ▼
                                          salvamento individual
                                                   │
                                                   ▼
                                               SHARDS
```

---

# 2. O que acontece durante o processamento?

O processamento possui quatro etapas principais:

```text
1. Carregar Parquet em RAM
              ↓
2. Ler TSV em streaming
              ↓
3. Tokenizar em batches e em paralelo
              ↓
4. Persistir cada exemplo dentro de shards
```

A ideia importante é que **batch e shard são coisas diferentes**.

- **Batch** determina como o trabalho é enviado para os workers.
- **Shard** determina como os resultados são armazenados no disco.

---

# 3. Parquet em RAM

O arquivo:

```text
analise_textos_split.parquet
```

é carregado inteiro em memória.

Ele contém informações como:

```text
dataset_id
index
split
```

Isso permite acessar rapidamente os metadados enquanto o TSV é processado.

O Parquet não precisa ser lido novamente a cada exemplo.

---

# 4. TSV em streaming

O TSV continua sendo lido linha por linha.

Isso é importante porque o TSV é grande demais para ser carregado inteiro em RAM.

O programa faz:

```text
abre TSV
   ↓
lê linha
   ↓
compara com o Parquet
   ↓
usa os textos
   ↓
continua para a próxima linha
```

O programa também verifica se:

```text
dataset_id do TSV == dataset_id do Parquet
index do TSV      == index do Parquet
```

Se houver uma diferença, o processamento é interrompido para evitar gerar dados incorretos.

---

# 5. Tokenização em batches

A tokenização continua sendo feita em batches.

Atualmente:

```python
PARQUET_BATCH_SIZE = 10_000
```

Isso significa que o programa monta algo como:

```text
Batch 1
10.000 exemplos
      ↓
Worker

Batch 2
10.000 exemplos
      ↓
Worker

Batch 3
10.000 exemplos
      ↓
Worker
```

Vários batches podem estar sendo processados simultaneamente pelos diferentes processos.

O número máximo de batches pendentes é controlado por:

```python
MAX_PENDING = 16
```

E o número de processos por:

```python
MAX_WORKERS = 16
```

---

# 6. O que é um shard?

Um **shard** é simplesmente um arquivo que funciona como um grande container para vários exemplos.

Em vez de fazer:

```text
exemplo_00000001.npz
exemplo_00000002.npz
exemplo_00000003.npz
exemplo_00000004.npz
...
```

fazemos:

```text
shard_000000.bin
shard_000001.bin
shard_000002.bin
...
```

Cada shard contém muitos exemplos.

Neste código:

```python
SHARD_SIZE = 100_000
```

Portanto:

```text
shard_000000.bin
    ├── exemplo 0
    ├── exemplo 1
    ├── exemplo 2
    ├── ...
    └── exemplo 99.999

shard_000001.bin
    ├── exemplo 100.000
    ├── exemplo 100.001
    ├── ...
```

O número de arquivos passa a ser muito menor.

---

# 7. O exemplo continua sendo salvo individualmente

Essa é a parte mais importante.

"Salvar em shard" **não significa esperar o shard ficar completo para salvar**.

Imagine que o batch terminou:

```text
Batch de 10.000 exemplos
```

O programa recebe os resultados e faz:

```text
Exemplo 1  → grava no shard
Exemplo 2  → grava no shard
Exemplo 3  → grava no shard
Exemplo 4  → grava no shard
...
Exemplo 10.000 → grava no shard
```

Ou seja, cada exemplo é persistido individualmente.

O shard é apenas o **container físico** onde esses exemplos são colocados.

---

# 8. Como um exemplo é armazenado?

Cada exemplo possui:

```text
src_tokens
tgt_tokens
src_length
tgt_length
```

Os tokens são armazenados em formato binário.

Como `MAX_SRC_LENGTH` e `MAX_TGT_LENGTH` são fixos, cada registro possui exatamente o mesmo tamanho.

Atualmente:

```python
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 256
```

Os tokens são armazenados como `int32`.

Isso permite calcular diretamente onde um exemplo está no arquivo:

```text
offset = índice_do_exemplo × tamanho_do_registro
```

Portanto, não é necessário criar um arquivo para cada exemplo.

---

# 9. E o JSONL?

Além do `.bin`, cada shard possui um `.jsonl`.

Por exemplo:

```text
shard_000123.bin
shard_000123.jsonl
```

O `.bin` contém os tokens.

O `.jsonl` contém os metadados.

Uma linha pode representar algo como:

```json
{
  "global_row": 12345678,
  "dataset_id": "algum_dataset_en_pt",
  "index": 54321,
  "split": "train",
  "shard": 123,
  "shard_index": 45678,
  "shard_path": ".../shard_000123.bin",
  "src_length": 42,
  "tgt_length": 37
}
```

Assim sabemos exatamente:

```text
qual exemplo é
        ↓
em qual shard está
        ↓
qual posição ocupa dentro do shard
        ↓
quais são seus comprimentos
```

---

# 10. Por que não criar um `.npz` por exemplo?

Para um dataset com dezenas de milhões de exemplos, isso seria problemático.

Por exemplo, para:

```text
51.000.000 exemplos
```

teríamos aproximadamente:

```text
51.000.000 arquivos
```

Além do espaço ocupado pelos dados, o sistema operacional precisa administrar:

- diretórios;
- inodes;
- permissões;
- timestamps;
- abertura/fechamento de arquivos;
- operações de filesystem;
- listagens;
- verificações;
- backups;
- exclusões.

Mesmo que cada arquivo seja pequeno, **dezenas de milhões de arquivos é uma situação muito ruim para um filesystem convencional**.

Com shards de 100.000 exemplos:

```text
51.000.000 / 100.000
≈ 510 shards
```

Em vez de aproximadamente:

```text
51 milhões de arquivos
```

passamos para aproximadamente:

```text
510 arquivos .bin
+ 510 arquivos .jsonl
```

fora os arquivos auxiliares.

---

# 11. Então os shards deixam a tokenização mais rápida?

## Não diretamente.

Essa distinção é muito importante.

O shard **não torna a operação de tokenização de um texto magicamente mais rápida**.

A tokenização continua sendo:

```text
texto
 ↓
tokenizer
 ↓
tokens
```

O ganho principal está na **persistência e no gerenciamento dos dados**.

---

# 12. Onde está o ganho?

O problema da abordagem anterior era principalmente o número gigantesco de arquivos.

Imagine:

```text
tokeniza
 ↓
cria exemplo.npz
 ↓
escreve arquivo
 ↓
fecha arquivo
 ↓
cria metadata.parquet
 ↓
fecha arquivo
 ↓
próximo exemplo
```

Repetido dezenas de milhões de vezes.

Isso gera uma quantidade enorme de operações de filesystem.

Com shards:

```text
abre shard
    ↓
escreve exemplo 1
    ↓
escreve exemplo 2
    ↓
escreve exemplo 3
    ↓
...
    ↓
escreve exemplo 100.000
    ↓
fecha shard
```

O filesystem passa a lidar com uma quantidade muito menor de arquivos.

---

# 13. Outra vantagem: retomada

Imagine que o programa esteja processando:

```text
exemplo 12.345.678
```

e o computador seja desligado.

Na abordagem de um arquivo por exemplo, existem milhões de arquivos individuais para verificar.

Com shards, o programa verifica o último shard.

Por exemplo:

```text
shard_000123.bin
shard_000123.jsonl
```

Ele compara:

```text
quantidade de registros no .bin
        versus
quantidade de registros no .jsonl
```

Se houve uma interrupção no meio de uma gravação, ele consegue truncar o registro incompleto e continuar.

---

# 14. O checkpoint

O arquivo:

```text
checkpoint.json
```

guarda informações como:

```json
{
  "tsv_rows_consumed": 12345678,
  "processed_examples": 12345678
}
```

O checkpoint só é atualizado depois que o batch correspondente foi persistido.

Assim:

```text
tokeniza batch
      ↓
grava exemplos
      ↓
batch persistido
      ↓
atualiza checkpoint
```

Isso evita que o checkpoint diga que um exemplo foi processado quando ele ainda não foi salvo.

---

# 15. O que acontece se o programa cair?

Suponha:

```text
Batch = 10.000
```

O programa terminou a tokenização e começou a gravar:

```text
1
2
3
...
7.352
```

e caiu.

Na próxima execução:

```text
abre os shards
      ↓
descobre o último registro válido
      ↓
remove eventual registro incompleto
      ↓
descobre a posição real
      ↓
avança o TSV até essa posição
      ↓
continua
```

Portanto, não é necessário refazer tudo desde o começo.

---

# 16. Por que o tamanho do registro é fixo?

Temos:

```python
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 256
```

Então cada exemplo sempre ocupa:

```text
256 tokens source
+
256 tokens target
+
2 comprimentos
```

Isso permite acesso direto.

Por exemplo:

```text
registro 0
registro 1
registro 2
registro 3
...
```

Se quisermos o registro 50.000:

```text
offset = 50.000 × RECORD_SIZE
```

O sistema pode ir diretamente para essa posição.

Isso é muito interessante para o treinamento posteriormente.

---

# 17. Fluxo completo

O processamento pode ser visualizado assim:

```text
                     ┌─────────────────────┐
                     │ Parquet             │
                     │ inteiro em RAM      │
                     └──────────┬──────────┘
                                │
                                │ metadados
                                ▼
┌─────────────┐          ┌───────────────┐
│ TSV         │─────────>│ Batch 10.000  │
│ streaming   │          └───────┬───────┘
└─────────────┘                  │
                                 ▼
                       ┌──────────────────┐
                       │ ProcessPool      │
                       │ 16 workers       │
                       └────────┬─────────┘
                                │
                                │ tokens
                                ▼
                       ┌──────────────────┐
                       │ Commit ordenado  │
                       └────────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
             exemplo 1                 exemplo 2
                    │                       │
                    └───────────┬───────────┘
                                ▼
                         ┌─────────────┐
                         │   SHARD     │
                         │ 100.000     │
                         │ exemplos    │
                         └──────┬──────┘
                                │
                         ┌──────┴──────┐
                         ▼             ▼
                    .bin tokens    .jsonl metadata
```

---

# 18. O que mudou em relação ao código anterior?

### Antes

A ideia era:

```text
batch
 ↓
tokenização
 ↓
exemplos individuais
 ↓
1 NPZ por exemplo
 ↓
1 Parquet por exemplo
```

Problema:

```text
51 milhões de exemplos
≈
51 milhões de arquivos NPZ
+
51 milhões de arquivos Parquet
```

Isso é extremamente pesado para o filesystem.

### Agora

Temos:

```text
batch
 ↓
tokenização
 ↓
exemplos individuais
 ↓
shard
```

Resultado aproximado para 51 milhões de exemplos:

```text
≈ 516 shards de 100.000 exemplos
```

mais os arquivos JSONL correspondentes.

---

# 19. O que NÃO mudou?

É importante deixar claro que a arquitetura de processamento continua sendo:

```text
Parquet → RAM
TSV → streaming
       ↓
batches
       ↓
ProcessPool
       ↓
tokenização paralela
```

Não transformamos a tokenização em um processamento de um exemplo por vez.

O batch continua existindo justamente para manter o throughput dos workers.

A mudança principal foi **onde e como os resultados são persistidos**.

---

# 20. Resumo

| Componente | Como funciona |
|---|---|
| TSV | Streaming |
| Parquet | Inteiro em RAM |
| Tokenização | Em batches |
| Paralelização | ProcessPool |
| Batch | 10.000 exemplos |
| Workers | 16 |
| Salvamento | Individual |
| Container | Shard |
| Exemplos por shard | 100.000 |
| Tokens | `.bin` |
| Metadata | `.jsonl` |
| Checkpoint | `checkpoint.json` |
| Retomada | Automática |
| Arquivo por exemplo | Não |
| Perda em uma interrupção | Muito menor |

## Em uma frase

**Batch é a unidade de processamento; exemplo é a unidade de persistência; shard é a unidade física de armazenamento.**

Essa separação é o principal motivo de termos escolhido essa arquitetura.
