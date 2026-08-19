import os
import gc
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from tokenizers import Tokenizer


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TSV_PATH = "/media/alvarinho/dados/Datasets/refined/traducao/analise_textos.tsv"
SPLIT_PARQUET_PATH = "/media/alvarinho/dados/Datasets/refined/traducao/analise_textos_split.parquet"

# Novo dataset tokenizado
OUTPUT_DIR = "/media/alvarinho/dados/Datasets/refined/traducao/tokenized"

# Parquet com a referência dos NPZs
OUTPUT_SPLIT_PARQUET = os.path.join(
    OUTPUT_DIR,
    "analise_textos_tokenized_split.parquet"
)

# Tokenizer já treinado
TOKENIZER_PATH = "artifacts/tokenizer_en_pt_es_120000.json"

# Comprimento máximo
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 256

# Quantidade de exemplos por NPZ
NPZ_CHUNK_SIZE = 100_000

# Número de linhas lidas do Parquet por vez
PARQUET_BATCH_SIZE = 1_000_000

# Número de linhas processadas do TSV por vez
TSV_BATCH_SIZE = 10_000


# ============================================================
# CARREGA OS ÍNDICES DO SPLIT
# ============================================================

def load_split_indices(parquet_path, batch_size=1_000_000):
    """
    Carrega dataset_id + index + split do Parquet original.

    Retorna:
        {
            "train": {
                dataset_id: set(indices),
                ...
            },
            "valid": {...},
            "test": {...}
        }
    """

    print("Carregando índices do split...")

    parquet_file = pq.ParquetFile(parquet_path)

    split_indices = {
        "train": {},
        "valid": {},
        "test": {}
    }

    total_rows = parquet_file.metadata.num_rows

    with tqdm(
        total=total_rows,
        desc="Mapeando split",
        unit="linhas"
    ) as pbar:

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["dataset_id", "index", "split"]
        ):

            rows = batch.num_rows
            df = batch.to_pandas()

            for split_name, group_split in df.groupby("split", observed=True):

                if split_name not in split_indices:
                    split_indices[split_name] = {}

                for dataset_id, group_dataset in group_split.groupby(
                    "dataset_id",
                    observed=True
                ):
                    split_indices[split_name].setdefault(dataset_id, set())

                    split_indices[split_name][dataset_id].update(
                        group_dataset["index"].astype(np.int64).tolist()
                    )

            del df
            gc.collect()

            pbar.update(rows)

    print("\nDistribuição encontrada:")

    for split_name, datasets in split_indices.items():

        total = sum(len(indices) for indices in datasets.values())

        print(
            f"  - {split_name}: {total:,}".replace(",", ".")
        )

        for dataset_id, indices in datasets.items():
            print(
                f"      {dataset_id}: {len(indices):,}".replace(",", ".")
            )

    return split_indices


# ============================================================
# PADDING
# ============================================================

def pad_tokens(tokens, max_length, pad_id):
    """
    Trunca e faz padding de uma sequência.

    Retorna:
        padded_tokens
        original_length após truncamento
    """

    # Mantém espaço para EOS caso a sequência ultrapasse o limite.
    tokens = tokens[:max_length]

    length = len(tokens)

    if length < max_length:
        tokens = tokens + [pad_id] * (max_length - length)

    return tokens, length


# ============================================================
# CRIAÇÃO DOS NPZS
# ============================================================

class NPZWriter:
    """
    Acumula exemplos e grava NPZs em chunks.

    Cada NPZ contém:

        src_tokens  -> (N, MAX_SRC_LENGTH)
        tgt_tokens  -> (N, MAX_TGT_LENGTH)
        src_lengths -> (N,)
        tgt_lengths -> (N,)
    """

    def __init__(
        self,
        output_dir,
        split_name,
        chunk_size,
        src_length,
        tgt_length,
        pad_id
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.split_name = split_name
        self.chunk_size = chunk_size

        self.src_length = src_length
        self.tgt_length = tgt_length
        self.pad_id = pad_id

        self.src_tokens = []
        self.tgt_tokens = []
        self.src_lengths = []
        self.tgt_lengths = []

        self.metadata = []

        self.chunk_id = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def add(
        self,
        dataset_id,
        index,
        src_tokens,
        tgt_tokens
    ):

        npz_index = len(self.src_tokens)

        src_padded, src_length = pad_tokens(
            src_tokens,
            self.src_length,
            self.pad_id
        )

        tgt_padded, tgt_length = pad_tokens(
            tgt_tokens,
            self.tgt_length,
            self.pad_id
        )

        self.src_tokens.append(src_padded)
        self.tgt_tokens.append(tgt_padded)

        self.src_lengths.append(src_length)
        self.tgt_lengths.append(tgt_length)

        self.metadata.append(
            {
                "dataset_id": dataset_id,
                "index": int(index),
                "split": self.split_name,
                "npz_index": npz_index
            }
        )

        if len(self.src_tokens) >= self.chunk_size:
            return self.flush()

        return []

    def flush(self):

        if not self.src_tokens:
            return []

        filename = (
            f"{self.split_name}_{self.chunk_id:05d}.npz"
        )

        npz_path = os.path.abspath(
            os.path.join(self.output_dir, filename)
        )

        np.savez(
            npz_path,
            src_tokens=np.asarray(
                self.src_tokens,
                dtype=np.int32
            ),
            tgt_tokens=np.asarray(
                self.tgt_tokens,
                dtype=np.int32
            ),
            src_lengths=np.asarray(
                self.src_lengths,
                dtype=np.int32
            ),
            tgt_lengths=np.asarray(
                self.tgt_lengths,
                dtype=np.int32
            )
        )

        print(
            f"\nSalvo: {npz_path}"
        )

        print(
            f"  exemplos: {len(self.src_tokens):,}".replace(",", ".")
        )

        print(
            f"  src shape: "
            f"({len(self.src_tokens)}, {self.src_length})"
        )

        print(
            f"  tgt shape: "
            f"({len(self.tgt_tokens)}, {self.tgt_length})"
        )

        # Adiciona o caminho completo ao metadata
        for row in self.metadata:
            row["npz_path"] = npz_path

        metadata = self.metadata

        self.src_tokens = []
        self.tgt_tokens = []
        self.src_lengths = []
        self.tgt_lengths = []
        self.metadata = []

        self.chunk_id += 1

        gc.collect()

        return metadata

    def close(self):
        return self.flush()


# ============================================================
# PROCESSAMENTO DO TSV
# ============================================================

def process_split(
    tsv_path,
    split_name,
    split_indices,
    tokenizer,
    output_dir,
    max_src_length,
    max_tgt_length,
    chunk_size,
    pad_id
):
    """
    Lê o TSV uma única vez para o split solicitado,
    tokeniza os exemplos e cria os NPZs.

    Retorna uma lista de metadados para o novo Parquet.
    """

    print()
    print("=" * 70)
    print(f"PROCESSANDO SPLIT: {split_name}")
    print("=" * 70)

    total_expected = sum(
        len(indices)
        for indices in split_indices.values()
    )

    print(
        f"Exemplos esperados: "
        f"{total_expected:,}".replace(",", ".")
    )

    writer = NPZWriter(
        output_dir=output_dir,
        split_name=split_name,
        chunk_size=chunk_size,
        src_length=max_src_length,
        tgt_length=max_tgt_length,
        pad_id=pad_id
    )

    metadata = []

    processed = 0
    skipped = 0

    with open(tsv_path, "r", encoding="utf-8") as f:

        # Cabeçalho
        next(f)

        with tqdm(
            total=total_expected,
            desc=f"Tokenizando {split_name}",
            unit="exemplos"
        ) as pbar:

            for line in f:

                line = line.rstrip("\n")

                try:
                    meta_part, texts_part = line.split(
                        "<METADATA>",
                        1
                    )

                    dataset_id, idx_str = meta_part.split(
                        "<SEP>",
                        1
                    )

                    text1, text2 = texts_part.split(
                        "<SEP>",
                        1
                    )

                    idx = int(idx_str)

                except ValueError:
                    skipped += 1
                    continue

                dataset_set = split_indices.get(dataset_id)

                if dataset_set is None:
                    continue

                if idx not in dataset_set:
                    continue

                # ------------------------------------------------
                # TAG DE DIREÇÃO
                # ------------------------------------------------

                if dataset_id.endswith("_en_pt"):
                    text1 = f"<2pt> {text1}"

                elif dataset_id.endswith("_en_es"):
                    text1 = f"<2es> {text1}"

                # ------------------------------------------------
                # TOKENIZAÇÃO
                # ------------------------------------------------

                encoded = tokenizer.encode(
                    text1,
                    text2
                )

                ids = encoded.ids

                # ------------------------------------------------
                # IMPORTANTE:
                #
                # O tokenizer está configurado com:
                #
                # <BOS> $A <SEP> $B <EOS>
                #
                # Portanto precisamos separar src/tgt.
                #
                # Para preservar exatamente source/target,
                # tokenizamos individualmente.
                # ------------------------------------------------

                src_encoded = tokenizer.encode(text1)
                tgt_encoded = tokenizer.encode(text2)

                src_ids = src_encoded.ids
                tgt_ids = tgt_encoded.ids

                rows = writer.add(
                    dataset_id=dataset_id,
                    index=idx,
                    src_tokens=src_ids,
                    tgt_tokens=tgt_ids
                )

                if rows:
                    metadata.extend(rows)

                processed += 1
                pbar.update(1)

    rows = writer.close()

    if rows:
        metadata.extend(rows)

    print()
    print(f"Processados: {processed:,}".replace(",", "."))
    print(f"Ignorados/malformados: {skipped:,}".replace(",", "."))

    return metadata


# ============================================================
# SALVA NOVO PARQUET
# ============================================================

def save_split_parquet(metadata, output_path):

    print()
    print("Salvando novo dataset de split...")

    if not metadata:
        raise RuntimeError(
            "Nenhum exemplo foi produzido."
        )

    df = pd.DataFrame(metadata)

    # Ordenação útil para leitura posterior
    df = df[
        [
            "dataset_id",
            "index",
            "split",
            "npz_path",
            "npz_index"
        ]
    ]

    # Garante tipos consistentes
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["index"] = df["index"].astype(np.int64)
    df["split"] = df["split"].astype(str)
    df["npz_path"] = df["npz_path"].astype(str)
    df["npz_index"] = df["npz_index"].astype(np.int64)

    os.makedirs(
        os.path.dirname(os.path.abspath(output_path)),
        exist_ok=True
    )

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    pq.write_table(
        table,
        output_path,
        compression="zstd"
    )

    print(f"Parquet salvo em:")
    print(f"  {os.path.abspath(output_path)}")

    print()
    print("Colunas:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print(
        f"Total de exemplos: "
        f"{len(df):,}".replace(",", ".")
    )

    print("\nDistribuição por split:")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TOKENIZAÇÃO + NPZ + NOVO DATASET DE SPLIT")
    print("=" * 70)

    os.makedirs(
        os.path.abspath(OUTPUT_DIR),
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Carrega tokenizer
    # --------------------------------------------------------

    print("\nCarregando tokenizer...")

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    pad_id = tokenizer.token_to_id("<PAD>")

    if pad_id is None:
        raise RuntimeError(
            "O tokenizer não possui o token <PAD>."
        )

    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"PAD ID: {pad_id}")
    print(f"MAX_SRC_LENGTH: {MAX_SRC_LENGTH}")
    print(f"MAX_TGT_LENGTH: {MAX_TGT_LENGTH}")
    print(f"NPZ_CHUNK_SIZE: {NPZ_CHUNK_SIZE:,}".replace(",", "."))

    # --------------------------------------------------------
    # 2. Carrega split original
    # --------------------------------------------------------

    split_indices = load_split_indices(
        SPLIT_PARQUET_PATH,
        batch_size=PARQUET_BATCH_SIZE
    )

    # --------------------------------------------------------
    # 3. Processa todos os splits
    # --------------------------------------------------------

    all_metadata = []

    for split_name in ["train", "valid", "test"]:

        current_indices = split_indices.get(
            split_name,
            {}
        )

        if not current_indices:
            print(
                f"\nNenhum exemplo encontrado para "
                f"'{split_name}'. Pulando."
            )
            continue

        metadata = process_split(
            tsv_path=TSV_PATH,
            split_name=split_name,
            split_indices=current_indices,
            tokenizer=tokenizer,
            output_dir=OUTPUT_DIR,
            max_src_length=MAX_SRC_LENGTH,
            max_tgt_length=MAX_TGT_LENGTH,
            chunk_size=NPZ_CHUNK_SIZE,
            pad_id=pad_id
        )

        all_metadata.extend(metadata)

        del metadata
        gc.collect()

    # --------------------------------------------------------
    # 4. Novo Parquet
    # --------------------------------------------------------

    save_split_parquet(
        metadata=all_metadata,
        output_path=OUTPUT_SPLIT_PARQUET
    )

    # --------------------------------------------------------
    # 5. Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

    print(f"\nNPZs:")
    print(f"  {os.path.abspath(OUTPUT_DIR)}")

    print(f"\nNovo split:")
    print(f"  {os.path.abspath(OUTPUT_SPLIT_PARQUET)}")

    print()
    print("Estrutura dos NPZs:")
    print("  src_tokens  -> tokens source já padded")
    print("  tgt_tokens  -> tokens target já padded")
    print("  src_lengths -> comprimento real da source")
    print("  tgt_lengths -> comprimento real do target")

    print()
    print("Estrutura do Parquet:")
    print("  dataset_id")
    print("  index")
    print("  split")
    print("  npz_path")
    print("  npz_index")