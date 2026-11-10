"""
Tokenização de tradução em blocos binários + um único Parquet de metadados.

Formato:
- Cada bloco possui dois arquivos binários: source e target.
- Os tokens são armazenados como int32, concatenados sequencialmente.
- O Parquet guarda block_id, row_in_block, offsets e comprimentos.
- O número de blocos nunca ultrapassa --max_blocks.
- Os arquivos .bin podem ser lidos com numpy.memmap, sem carregar o bloco inteiro.

Dependências:
    pip install numpy pandas pyarrow tokenizers tqdm

ATENÇÃO:
- Por padrão, o script começa do zero e apaga OUTPUT_DIR.
- Só mantenha CLEAN_OUTPUT_DIR_AT_START=True se OUTPUT_DIR for exclusivamente
  a saída deste pipeline.
"""

import os
import math
import shutil
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TSV_PATH = (
    "/media/alvarinho/dados/Datasets/refined/traducao/"
    "analise_textos.tsv"
)
SPLIT_PARQUET_PATH = (
    "/media/alvarinho/dados/Datasets/refined/traducao/"
    "analise_textos_split.parquet"
)
TOKENIZER_PATH = "artifacts/tokenizer_en_pt_es_60000.json"

OUTPUT_DIR = (
    "/media/alvarinho/dados/Datasets/refined/traducao/"
    "tokenized"
)
BLOCKS_DIR = os.path.join(OUTPUT_DIR, "blocks")
OUTPUT_METADATA_PARQUET = os.path.join(
    OUTPUT_DIR, "analise_textos_tokenized.parquet"
)

BATCH_SIZE = 20_000
MAX_WORKERS = min(16, os.cpu_count() or 1)

# Número máximo de blocos. Também pode ser informado por linha de comando:
# python tokenize.py --max_blocks 128
DEFAULT_MAX_BLOCKS = 128

CLEAN_OUTPUT_DIR_AT_START = True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ============================================================
# TOKENIZAÇÃO NOS WORKERS
# ============================================================

_WORKER_TOKENIZER = None


def init_worker(tokenizer_path):
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = Tokenizer.from_file(tokenizer_path)


def tokenize_batch(rows):
    """
    rows:
      (global_row, dataset_id, index, split, src_text, tgt_text)
    """
    output = []

    for global_row, dataset_id, index, split, src_text, tgt_text in rows:
        if dataset_id.endswith("_en_pt"):
            tgt_text = f"<2pt> {tgt_text}"
        elif dataset_id.endswith("_en_es"):
            tgt_text = f"<2es> {tgt_text}"

        src_ids = _WORKER_TOKENIZER.encode(src_text).ids
        tgt_ids = _WORKER_TOKENIZER.encode(tgt_text).ids

        output.append({
            "global_row": int(global_row),
            "dataset_id": str(dataset_id),
            "index": int(index),
            "split": str(split),
            "src_tokens": np.asarray(src_ids, dtype=np.int32),
            "tgt_tokens": np.asarray(tgt_ids, dtype=np.int32),
            "src_length": len(src_ids),
            "tgt_length": len(tgt_ids),
        })

    return output


# ============================================================
# PARSE TSV
# ============================================================

def parse_tsv_line(line, line_number):
    line = line.rstrip("\r\n")
    try:
        meta_part, texts_part = line.split("<METADATA>", 1)
        dataset_id, index_str = meta_part.split("<SEP>", 1)
        src_text, tgt_text = texts_part.split("<SEP>", 1)
        return str(dataset_id), int(index_str), src_text, tgt_text
    except ValueError as exc:
        raise ValueError(
            f"Linha TSV malformada na linha {line_number}: {line[:200]!r}"
        ) from exc


# ============================================================
# METADADOS
# ============================================================

METADATA_SCHEMA = pa.schema([
    ("global_row", pa.int64()),
    ("dataset_id", pa.string()),
    ("index", pa.int64()),
    ("split", pa.string()),
    ("block_id", pa.int32()),
    ("row_in_block", pa.int32()),
    # Offsets medidos em quantidade de elementos int32, não em bytes.
    ("src_offset", pa.int64()),
    ("tgt_offset", pa.int64()),
    ("src_length", pa.int32()),
    ("tgt_length", pa.int32()),
])


class BlockWriter:
    """Grava tokens sequencialmente em arquivos .bin por bloco."""

    def __init__(self, blocks_dir, rows_per_block):
        self.blocks_dir = blocks_dir
        self.rows_per_block = int(rows_per_block)

        self.block_id = 0
        self.row_in_block = 0
        self.src_offset = 0
        self.tgt_offset = 0
        self.src_file = None
        self.tgt_file = None

        os.makedirs(blocks_dir, exist_ok=True)

    def _open_block(self):
        prefix = os.path.join(self.blocks_dir, f"block_{self.block_id:05d}")
        self.src_file = open(prefix + "_src.bin", "wb")
        self.tgt_file = open(prefix + "_tgt.bin", "wb")

    def _close_block(self):
        if self.src_file is not None:
            self.src_file.close()
            self.tgt_file.close()
            self.src_file = None
            self.tgt_file = None

    def write(self, record):
        if self.row_in_block >= self.rows_per_block:
            self._close_block()
            self.block_id += 1
            self.row_in_block = 0
            self.src_offset = 0
            self.tgt_offset = 0

        if self.src_file is None:
            self._open_block()

        src = np.asarray(record["src_tokens"], dtype=np.int32)
        tgt = np.asarray(record["tgt_tokens"], dtype=np.int32)

        metadata = {
            "global_row": record["global_row"],
            "dataset_id": record["dataset_id"],
            "index": record["index"],
            "split": record["split"],
            "block_id": self.block_id,
            "row_in_block": self.row_in_block,
            "src_offset": self.src_offset,
            "tgt_offset": self.tgt_offset,
            "src_length": len(src),
            "tgt_length": len(tgt),
        }

        src.tofile(self.src_file)
        tgt.tofile(self.tgt_file)

        self.src_offset += len(src)
        self.tgt_offset += len(tgt)
        self.row_in_block += 1

        return metadata

    def close(self):
        self._close_block()


def write_metadata_batch(writer, metadata_rows):
    if metadata_rows:
        table = pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA)
        writer.write_table(table, row_group_size=len(metadata_rows))


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_blocks",
        type=int,
        default=DEFAULT_MAX_BLOCKS,
        help="Número máximo de blocos binários de tokens.",
    )
    args = parser.parse_args()

    if args.max_blocks < 1:
        raise ValueError("--max_blocks precisa ser maior ou igual a 1.")

    if CLEAN_OUTPUT_DIR_AT_START and os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(BLOCKS_DIR, exist_ok=True)

    if not os.path.isfile(TSV_PATH):
        raise FileNotFoundError(f"TSV não encontrado: {TSV_PATH}")
    if not os.path.isfile(SPLIT_PARQUET_PATH):
        raise FileNotFoundError(
            f"Parquet de splits não encontrado: {SPLIT_PARQUET_PATH}"
        )
    if not os.path.isfile(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer não encontrado: {TOKENIZER_PATH}")

    print("Lendo metadados de split...")
    split_df = pd.read_parquet(
        SPLIT_PARQUET_PATH,
        columns=["dataset_id", "index", "split"],
    )
    total_rows = len(split_df)
    print(f"Exemplos: {total_rows:,}".replace(",", "."))

    if total_rows == 0:
        raise ValueError("O Parquet de splits não contém exemplos.")

    # ceil garante que o total de blocos seja <= max_blocks.
    rows_per_block = math.ceil(total_rows / args.max_blocks)
    expected_blocks = math.ceil(total_rows / rows_per_block)

    print(f"Máximo de blocos solicitado: {args.max_blocks}")
    print(f"Exemplos por bloco: aproximadamente {rows_per_block:,}")
    print(f"Quantidade efetiva de blocos: {expected_blocks}")

    # Valida que o tokenizer pode ser aberto antes de iniciar os workers.
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    del tokenizer

    metadata_writer = pq.ParquetWriter(
        OUTPUT_METADATA_PARQUET,
        METADATA_SCHEMA,
        compression="zstd",
        use_dictionary=["dataset_id", "split"],
    )

    block_writer = BlockWriter(BLOCKS_DIR, rows_per_block)

    try:
        with open(TSV_PATH, "r", encoding="utf-8", newline="") as tsv:
            header = next(tsv, None)
            if header is None:
                raise ValueError("O TSV está vazio.")

            ctx = mp.get_context("fork")

            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(TOKENIZER_PATH,),
            ) as pool:
                pbar = tqdm(total=total_rows, desc="Tokenizando", unit="ex")
                global_row = 0

                try:
                    while global_row < total_rows:
                        end = min(global_row + BATCH_SIZE, total_rows)
                        split_chunk = split_df.iloc[global_row:end]
                        rows = []

                        for offset, row in enumerate(
                            split_chunk.itertuples(index=False)
                        ):
                            line_number = global_row + offset + 2
                            line = tsv.readline()

                            if not line:
                                raise RuntimeError(
                                    f"TSV terminou antes do Parquet, "
                                    f"na linha {line_number}."
                                )

                            dataset_id_tsv, index_tsv, src_text, tgt_text = (
                                parse_tsv_line(line, line_number)
                            )

                            if (
                                dataset_id_tsv != str(row.dataset_id)
                                or index_tsv != int(row.index)
                            ):
                                raise RuntimeError(
                                    "TSV e Parquet estão desalinhados na "
                                    f"posição {global_row + offset}: "
                                    f"TSV=({dataset_id_tsv}, {index_tsv}), "
                                    f"Parquet=({row.dataset_id}, {row.index})."
                                )

                            rows.append((
                                global_row + offset,
                                dataset_id_tsv,
                                index_tsv,
                                str(row.split),
                                src_text,
                                tgt_text,
                            ))

                        chunk_size = max(
                            1, (len(rows) + MAX_WORKERS - 1) // MAX_WORKERS
                        )
                        row_chunks = [
                            rows[i:i + chunk_size]
                            for i in range(0, len(rows), chunk_size)
                        ]

                        tokenized_chunks = pool.map(tokenize_batch, row_chunks)

                        metadata_rows = []
                        for tokenized_chunk in tokenized_chunks:
                            for record in tokenized_chunk:
                                metadata_rows.append(block_writer.write(record))

                        write_metadata_batch(metadata_writer, metadata_rows)

                        global_row = end
                        pbar.update(len(rows))

                        del (
                            rows,
                            row_chunks,
                            tokenized_chunks,
                            metadata_rows,
                            split_chunk,
                        )
                finally:
                    pbar.close()

            extra_line = tsv.readline()
            if extra_line:
                raise RuntimeError(
                    "O TSV possui linhas extras após o fim do Parquet de splits."
                )

    finally:
        block_writer.close()
        metadata_writer.close()

    del split_df

    print("\nConcluído.")
    print(f"Blocos de tokens: {BLOCKS_DIR}")
    print(f"Metadados: {OUTPUT_METADATA_PARQUET}")
    print(f"Quantidade máxima configurada: {args.max_blocks}")


if __name__ == "__main__":
    main()
