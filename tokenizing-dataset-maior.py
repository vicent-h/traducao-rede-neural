import os
import gc
import json
import time
import multiprocessing as mp
import math

from concurrent.futures import (
    ProcessPoolExecutor,
    wait,
    FIRST_COMPLETED,
)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
from tokenizers import Tokenizer


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

OUTPUT_DIR = (
    "/media/alvarinho/dados/Datasets/refined/traducao/"
    "tokenized"
)

OUTPUT_SPLIT_PARQUET = os.path.join(
    OUTPUT_DIR,
    "analise_textos_tokenized_split.parquet",
)

TOKENIZER_PATH = (
    "artifacts/tokenizer_en_pt_es_120000.json"
)


# ============================================================
# PARÂMETROS
# ============================================================

MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 256

# Exemplos por NPZ
NPZ_CHUNK_SIZE = 100_000

# Quantas linhas do Parquet/TSV são lidas por vez
PARQUET_BATCH_SIZE = 10_000

# Quantos batches podem ficar simultaneamente nos workers
MAX_PENDING = 16

# Número de processos de tokenização
MAX_WORKERS = 16


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "checkpoint.json",
)


# ============================================================
# CONFIGURAÇÃO DO PROCESSAMENTO
# ============================================================

CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "processing_config.json",
)


# ============================================================
# EVITA OVERSUBSCRIPTION
# ============================================================

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)


# ============================================================
# CHECKPOINT
# ============================================================

def atomic_write_json(path, data):
    """
    Escreve JSON de forma atômica.

    Nunca sobrescreve diretamente o checkpoint.
    """

    temp_path = path + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(
        temp_path,
        path,
    )


def load_checkpoint():
    """
    Carrega o checkpoint existente.

    Se não existir, começa do zero.
    """

    if not os.path.exists(
        CHECKPOINT_PATH
    ):
        return {
            "tsv_rows_consumed": 0,
            "processed_examples": 0,
            "train_chunks": 0,
            "val_chunks": 0,
            "test_chunks": 0,
        }

    print(
        f"\nCheckpoint encontrado:"
        f"\n  {CHECKPOINT_PATH}"
    )

    try:

        with open(
            CHECKPOINT_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            checkpoint = json.load(f)

        return checkpoint

    except Exception as exc:

        raise RuntimeError(
            "Não foi possível ler o checkpoint.\n"
            f"Arquivo: {CHECKPOINT_PATH}\n"
            f"Erro: {exc}"
        )


def save_checkpoint(
    tsv_rows_consumed,
    processed_examples,
    writers,
):
    """
    Salva checkpoint somente depois que os dados
    correspondentes já foram gravados com sucesso.
    """

    checkpoint = {
        "tsv_rows_consumed": int(
            tsv_rows_consumed
        ),

        "processed_examples": int(
            processed_examples
        ),

        "train_chunks": int(
            writers["train"].chunk_id
        ),

        "val_chunks": int(
            writers["val"].chunk_id
        ),

        "test_chunks": int(
            writers["test"].chunk_id
        ),

        "updated_at": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    atomic_write_json(
        CHECKPOINT_PATH,
        checkpoint,
    )


# ============================================================
# CONFIGURAÇÃO
# ============================================================

def get_processing_config():
    return {
        "tsv_path": os.path.abspath(
            TSV_PATH
        ),

        "split_parquet_path": os.path.abspath(
            SPLIT_PARQUET_PATH
        ),

        "tokenizer_path": os.path.abspath(
            TOKENIZER_PATH
        ),

        "max_src_length": MAX_SRC_LENGTH,
        "max_tgt_length": MAX_TGT_LENGTH,
        "npz_chunk_size": NPZ_CHUNK_SIZE,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
    }


def verify_processing_config():
    """
    Impede continuar um checkpoint com uma configuração
    diferente da utilizada originalmente.
    """

    current_config = get_processing_config()

    if not os.path.exists(
        CONFIG_PATH
    ):

        atomic_write_json(
            CONFIG_PATH,
            current_config,
        )

        return

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        old_config = json.load(f)

    if old_config != current_config:

        print("\nConfiguração anterior:")
        print(
            json.dumps(
                old_config,
                indent=2,
                ensure_ascii=False,
            )
        )

        print("\nConfiguração atual:")
        print(
            json.dumps(
                current_config,
                indent=2,
                ensure_ascii=False,
            )
        )

        raise RuntimeError(
            "\nA configuração mudou desde o último processamento.\n"
            "Para evitar corromper/duplicar o dataset, "
            "não continuarei automaticamente.\n\n"
            "Se realmente quiser começar do zero, remova:\n"
            f"  {CHECKPOINT_PATH}\n"
            f"  {CONFIG_PATH}\n"
            "e os NPZ/metadata gerados."
        )


# ============================================================
# PADDING
# ============================================================

def pad_tokens(
    tokens,
    max_length,
    pad_id,
):
    """
    Trunca e faz padding.
    """

    tokens = tokens[:max_length]

    length = len(tokens)

    if length < max_length:

        tokens = (
            tokens
            + [pad_id] * (
                max_length - length
            )
        )

    return tokens, length


# ============================================================
# NPZ WRITER
# ============================================================

class NPZWriter:

    def __init__(
        self,
        output_dir,
        split_name,
        chunk_size,
        src_length,
        tgt_length,
        pad_id,
    ):

        self.output_dir = os.path.abspath(
            output_dir
        )

        self.meta_parts_dir = os.path.join(
            self.output_dir,
            "metadata_parts",
        )

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

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        os.makedirs(
            self.meta_parts_dir,
            exist_ok=True,
        )

        self._detect_existing_chunks()

    # --------------------------------------------------------
    # Detecta chunks já existentes
    # --------------------------------------------------------

    def _detect_existing_chunks(self):

        chunk_ids = []

        prefix = (
            f"{self.split_name}_"
        )

        for filename in os.listdir(
            self.output_dir
        ):

            if not filename.startswith(
                prefix
            ):
                continue

            if not filename.endswith(
                ".npz"
            ):
                continue

            number_part = filename[
                len(prefix):-4
            ]

            try:

                chunk_id = int(
                    number_part
                )

                chunk_ids.append(
                    chunk_id
                )

            except ValueError:
                continue

        if chunk_ids:

            self.chunk_id = (
                max(chunk_ids) + 1
            )

    # --------------------------------------------------------
    # Add
    # --------------------------------------------------------

    def add(
        self,
        dataset_id,
        index,
        src_tokens,
        tgt_tokens,
    ):

        npz_index = len(
            self.src_tokens
        )

        src_padded, src_length = (
            pad_tokens(
                src_tokens,
                self.src_length,
                self.pad_id,
            )
        )

        tgt_padded, tgt_length = (
            pad_tokens(
                tgt_tokens,
                self.tgt_length,
                self.pad_id,
            )
        )

        self.src_tokens.append(
            src_padded
        )

        self.tgt_tokens.append(
            tgt_padded
        )

        self.src_lengths.append(
            src_length
        )

        self.tgt_lengths.append(
            tgt_length
        )

        self.metadata.append(
            {
                "dataset_id": dataset_id,
                "index": int(index),
                "split": self.split_name,
                "npz_index": npz_index,
            }
        )

        if (
            len(self.src_tokens)
            >= self.chunk_size
        ):

            return self.flush()

        return False

    # --------------------------------------------------------
    # Flush
    # --------------------------------------------------------

    def flush(self):

        if not self.src_tokens:
            return False

        chunk_id = self.chunk_id

        base_name = (
            f"{self.split_name}_"
            f"{chunk_id:05d}"
        )

        npz_filename = (
            base_name + ".npz"
        )

        meta_filename = (
            base_name
            + "_meta.parquet"
        )

        npz_path = os.path.abspath(
            os.path.join(
                self.output_dir,
                npz_filename,
            )
        )

        meta_path = os.path.abspath(
            os.path.join(
                self.meta_parts_dir,
                meta_filename,
            )
        )

        # ----------------------------------------------------
        # Arquivos temporários
        # ----------------------------------------------------

        npz_tmp = npz_path + ".tmp"
        meta_tmp = meta_path + ".tmp"

        # ----------------------------------------------------
        # Segurança
        # ----------------------------------------------------

        if os.path.exists(npz_tmp):
            os.remove(npz_tmp)

        if os.path.exists(meta_tmp):
            os.remove(meta_tmp)

        # ----------------------------------------------------
        # 1. Salva NPZ temporário
        # ----------------------------------------------------

        np.savez(
            npz_tmp,
            src_tokens=np.asarray(
                self.src_tokens,
                dtype=np.int32,
            ),
            tgt_tokens=np.asarray(
                self.tgt_tokens,
                dtype=np.int32,
            ),
            src_lengths=np.asarray(
                self.src_lengths,
                dtype=np.int32,
            ),
            tgt_lengths=np.asarray(
                self.tgt_lengths,
                dtype=np.int32,
            ),
        )

        # np.savez pode adicionar ".npz"
        if os.path.exists(
            npz_tmp + ".npz"
        ):

            os.replace(
                npz_tmp + ".npz",
                npz_tmp,
            )

        # ----------------------------------------------------
        # Garante que NPZ terminou
        # ----------------------------------------------------

        if not os.path.exists(
            npz_tmp
        ):

            raise RuntimeError(
                f"NPZ temporário não foi criado: "
                f"{npz_tmp}"
            )

        # ----------------------------------------------------
        # 2. Metadata
        # ----------------------------------------------------

        for row in self.metadata:

            row["npz_path"] = (
                npz_path
            )

        df_meta = pd.DataFrame(
            self.metadata
        )

        df_meta = df_meta[
            [
                "dataset_id",
                "index",
                "split",
                "npz_path",
                "npz_index",
            ]
        ]

        df_meta["dataset_id"] = (
            df_meta["dataset_id"]
            .astype(str)
        )

        df_meta["index"] = (
            df_meta["index"]
            .astype(np.int64)
        )

        df_meta["split"] = (
            df_meta["split"]
            .astype(str)
        )

        df_meta["npz_path"] = (
            df_meta["npz_path"]
            .astype(str)
        )

        df_meta["npz_index"] = (
            df_meta["npz_index"]
            .astype(np.int64)
        )

        table = pa.Table.from_pandas(
            df_meta,
            preserve_index=False,
        )

        pq.write_table(
            table,
            meta_tmp,
            compression="zstd",
        )

        del df_meta
        del table

        # ----------------------------------------------------
        # 3. Commit atômico
        # ----------------------------------------------------

        os.replace(
            npz_tmp,
            npz_path,
        )

        os.replace(
            meta_tmp,
            meta_path,
        )

        # ----------------------------------------------------
        # 4. Limpa RAM
        # ----------------------------------------------------

        count = len(
            self.src_tokens
        )

        self.src_tokens = []
        self.tgt_tokens = []

        self.src_lengths = []
        self.tgt_lengths = []

        self.metadata = []

        self.chunk_id += 1

        gc.collect()

        print(
            f"\n[{self.split_name}] "
            f"Chunk {chunk_id:05d} concluído "
            f"({count:,} exemplos)"
        )

        return True

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    def close(self):

        return self.flush()


# ============================================================
# WORKER GLOBAL
# ============================================================

_WORKER_TOKENIZER = None


# ============================================================
# WORKER INIT
# ============================================================

def init_worker(
    tokenizer_path,
):

    global _WORKER_TOKENIZER

    _WORKER_TOKENIZER = (
        Tokenizer.from_file(
            tokenizer_path
        )
    )


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenize_batch(rows):
    """
    rows:

        (
            split,
            dataset_id,
            index,
            text1,
            text2,
        )
    """

    results = []
    skipped = 0

    for (
        split_name,
        dataset_id,
        idx,
        text1,
        text2,
    ) in rows:

        # ----------------------------------------------------
        # Direção
        # ----------------------------------------------------

        if dataset_id.endswith(
            "_en_pt"
        ):

            text1 = (
                f"<2pt> {text1}"
            )

        elif dataset_id.endswith(
            "_en_es"
        ):

            text1 = (
                f"<2es> {text1}"
            )

        # ----------------------------------------------------
        # Tokenização
        # ----------------------------------------------------

        try:

            src_encoded = (
                _WORKER_TOKENIZER.encode(
                    text1
                )
            )

            tgt_encoded = (
                _WORKER_TOKENIZER.encode(
                    text2
                )
            )

        except Exception:

            skipped += 1
            continue

        results.append(
            (
                split_name,
                dataset_id,
                idx,
                src_encoded.ids,
                tgt_encoded.ids,
            )
        )

    return (
        results,
        skipped,
    )


# ============================================================
# PARSE TSV
# ============================================================

def parse_tsv_line(line):

    line = line.rstrip("\n")

    try:

        meta_part, texts_part = (
            line.split(
                "<METADATA>",
                1,
            )
        )

        dataset_id, idx_str = (
            meta_part.split(
                "<SEP>",
                1,
            )
        )

        text1, text2 = (
            texts_part.split(
                "<SEP>",
                1,
            )
        )

        idx = int(idx_str)

    except ValueError:

        return None

    return (
        dataset_id,
        idx,
        text1,
        text2,
    )


# ============================================================
# LEITURA DO PARQUET
# ============================================================

def iter_parquet_batches(
    parquet_path,
    start_row,
):

    parquet_file = pq.ParquetFile(
        parquet_path
    )

    total_rows = (
        parquet_file.metadata.num_rows
    )

    if start_row >= total_rows:

        return

    # --------------------------------------------------------
    # Para retomar do checkpoint sem carregar tudo
    # --------------------------------------------------------

    rows_to_skip = start_row

    for batch in parquet_file.iter_batches(
        batch_size=PARQUET_BATCH_SIZE,
        columns=[
            "dataset_id",
            "index",
            "split",
        ],
    ):

        if rows_to_skip > 0:

            if (
                rows_to_skip
                >= batch.num_rows
            ):

                rows_to_skip -= (
                    batch.num_rows
                )

                continue

            # --------------------------------------------
            # Estamos dentro deste batch
            # --------------------------------------------

            batch = batch.slice(
                rows_to_skip
            )

            rows_to_skip = 0

        df = batch.to_pandas()

        yield df

        del df


# ============================================================
# PROCESSAMENTO
# ============================================================

def process_dataset(
    tsv_path,
    parquet_path,
    output_dir,
    tokenizer_path,
    pad_id,
    parquet_rows
):

    checkpoint = (
        load_checkpoint()
    )

    start_row = int(
        checkpoint[
            "tsv_rows_consumed"
        ]
    )

    processed_examples = int(
        checkpoint[
            "processed_examples"
        ]
    )

    print()
    print("=" * 70)
    print("RETOMADA")
    print("=" * 70)

    print(
        f"Linhas já consumidas: "
        f"{start_row:,}".replace(
            ",",
            ".",
        )
    )

    print(
        f"Exemplos já processados: "
        f"{processed_examples:,}".replace(
            ",",
            ".",
        )
    )

    # --------------------------------------------------------
    # Writers
    # --------------------------------------------------------

    writers = {
        "train": NPZWriter(
            output_dir,
            "train",
            NPZ_CHUNK_SIZE,
            MAX_SRC_LENGTH,
            MAX_TGT_LENGTH,
            pad_id,
        ),

        "val": NPZWriter(
            output_dir,
            "val",
            NPZ_CHUNK_SIZE,
            MAX_SRC_LENGTH,
            MAX_TGT_LENGTH,
            pad_id,
        ),

        "test": NPZWriter(
            output_dir,
            "test",
            NPZ_CHUNK_SIZE,
            MAX_SRC_LENGTH,
            MAX_TGT_LENGTH,
            pad_id,
        ),
    }

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    workers = min(
        MAX_WORKERS,
        os.cpu_count() or 1,
    )

    print(
        f"Workers: {workers}"
    )

    mp_context = mp.get_context(
        "fork"
    )

    pending = set()

    # --------------------------------------------------------
    # Abre TSV
    # --------------------------------------------------------

    with open(
        tsv_path,
        "r",
        encoding="utf-8",
    ) as tsv:

        # --------------------------------------------
        # Cabeçalho
        # --------------------------------------------

        next(tsv)

        # --------------------------------------------
        # Avança TSV até checkpoint
        # --------------------------------------------

        if start_row > 0:

            print(
                "Avançando TSV até "
                "o checkpoint..."
            )

            skipped_lines = 0

            with tqdm(
                total=start_row,
                initial=0,
                desc="Recuperando posição",
                unit="linha",
            ) as pbar:

                while (
                    skipped_lines
                    < start_row
                ):

                    line = tsv.readline()

                    if not line:

                        raise RuntimeError(
                            "O TSV terminou antes "
                            "do ponto salvo no checkpoint."
                        )

                    skipped_lines += 1

                    if (
                        skipped_lines % 10_000
                        == 0
                    ):

                        pbar.update(
                            10_000
                        )

                remaining = (
                    skipped_lines
                    % 10_000
                )

                if remaining:
                    pbar.update(
                        remaining
                    )

    # ========================================================
    # REABRE TSV
    # ========================================================

    with open(
        tsv_path,
        "r",
        encoding="utf-8",
    ) as tsv:

        next(tsv)

        # --------------------------------------------
        # Posiciona novamente
        # --------------------------------------------

        for _ in range(
            start_row
        ):

            next(tsv)

        # --------------------------------------------
        # Executor
        # --------------------------------------------

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context,
            initializer=init_worker,
            initargs=(
                tokenizer_path,
            ),
        ) as executor:

            parquet_batches = (
                iter_parquet_batches(
                    parquet_path,
                    start_row,
                )
            )

            current_row = start_row

            # ====================================================
            # LOOP PRINCIPAL
            # ====================================================

            num_batches = math.ceil(
                parquet_rows / PARQUET_BATCH_SIZE
            )

            start_batch = start_row // PARQUET_BATCH_SIZE

            for parquet_df in tqdm(
                parquet_batches,
                desc="Processando",
                unit="batch",
                total=num_batches,
                initial=start_batch,
            ):

                batch_rows = []

                # ------------------------------------------------
                # Lê TSV exatamente na mesma quantidade de linhas
                # ------------------------------------------------

                for row in parquet_df.itertuples(
                    index=False
                ):

                    line = tsv.readline()

                    if not line:

                        raise RuntimeError(
                            "TSV terminou antes "
                            "do Parquet."
                        )

                    current_row += 1

                    parsed = (
                        parse_tsv_line(
                            line
                        )
                    )

                    if parsed is None:

                        raise RuntimeError(
                            "Linha malformada no TSV "
                            f"na posição {current_row}."
                        )

                    (
                        dataset_id_tsv,
                        index_tsv,
                        text1,
                        text2,
                    ) = parsed

                    # --------------------------------------------
                    # VERIFICAÇÃO CRÍTICA
                    # --------------------------------------------

                    if (
                        dataset_id_tsv
                        != row.dataset_id
                        or int(index_tsv)
                        != int(row.index)
                    ):

                        raise RuntimeError(
                            "\n\n"
                            "ERRO DE ORDEM!\n"
                            "O Parquet e o TSV não estão "
                            "na mesma ordem.\n\n"
                            f"TSV:\n"
                            f"  dataset_id = "
                            f"{dataset_id_tsv}\n"
                            f"  index      = "
                            f"{index_tsv}\n\n"
                            f"Parquet:\n"
                            f"  dataset_id = "
                            f"{row.dataset_id}\n"
                            f"  index      = "
                            f"{row.index}\n\n"
                            f"Posição: "
                            f"{current_row}\n\n"
                            "O processamento foi interrompido "
                            "para evitar gerar dados incorretos."
                        )

                    batch_rows.append(
                        (
                            row.split,
                            dataset_id_tsv,
                            index_tsv,
                            text1,
                            text2,
                        )
                    )

                del parquet_df

                # ------------------------------------------------
                # Envia batch ao worker
                # ------------------------------------------------

                if batch_rows:

                    future = (
                        executor.submit(
                            tokenize_batch,
                            batch_rows,
                        )
                    )

                    pending.add(
                        future
                    )

                    del batch_rows

                # ------------------------------------------------
                # Limita memória
                # ------------------------------------------------

                if (
                    len(pending)
                    >= MAX_PENDING
                ):

                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    # --------------------------------------------
                    # Consome resultados
                    # --------------------------------------------

                    for future in done:

                        results, skipped = (
                            future.result()
                        )

                        # ----------------------------------------
                        # Salva tokens
                        # ----------------------------------------

                        for (
                            split_name,
                            dataset_id,
                            idx,
                            src_tokens,
                            tgt_tokens,
                        ) in results:

                            writers[
                                split_name
                            ].add(
                                dataset_id,
                                idx,
                                src_tokens,
                                tgt_tokens,
                            )

                        processed_examples += (
                            len(results)
                        )

                        del results

                        # ----------------------------------------
                        # IMPORTANTE:
                        #
                        # checkpoint só deve avançar depois
                        # que os dados foram efetivamente gravados.
                        # ----------------------------------------

                    save_checkpoint(
                        tsv_rows_consumed=current_row,
                        processed_examples=processed_examples,
                        writers=writers,
                    )

                    gc.collect()

            # ====================================================
            # FINALIZA WORKERS
            # ====================================================

            while pending:

                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:

                    results, skipped = (
                        future.result()
                    )

                    for (
                        split_name,
                        dataset_id,
                        idx,
                        src_tokens,
                        tgt_tokens,
                    ) in results:

                        writers[
                            split_name
                        ].add(
                            dataset_id,
                            idx,
                            src_tokens,
                            tgt_tokens,
                        )

                    processed_examples += (
                        len(results)
                    )

                    del results

                save_checkpoint(
                    tsv_rows_consumed=current_row,
                    processed_examples=processed_examples,
                    writers=writers,
                )

    # ========================================================
    # FLUSH FINAL
    # ========================================================

    for writer in writers.values():

        writer.flush()

    # --------------------------------------------------------
    # Checkpoint final
    # --------------------------------------------------------

    save_checkpoint(
        tsv_rows_consumed=current_row,
        processed_examples=processed_examples,
        writers=writers,
    )

    print()
    print("=" * 70)
    print("TOKENIZAÇÃO FINALIZADA")
    print("=" * 70)

    print(
        f"Linhas consumidas: "
        f"{current_row:,}".replace(
            ",",
            ".",
        )
    )

    print(
        f"Exemplos processados: "
        f"{processed_examples:,}".replace(
            ",",
            ".",
        )
    )


# ============================================================
# CONSOLIDA METADADOS
# ============================================================

def consolidate_metadata(
    metadata_parts_dir,
    output_path,
):

    print()
    print("=" * 70)
    print("CONSOLIDANDO METADADOS")
    print("=" * 70)

    meta_files = sorted(
        os.path.join(
            metadata_parts_dir,
            filename,
        )
        for filename in os.listdir(
            metadata_parts_dir
        )
        if filename.endswith(
            "_meta.parquet"
        )
        and not filename.endswith(
            ".tmp"
        )
    )

    if not meta_files:

        raise RuntimeError(
            "Nenhum metadata Parquet encontrado."
        )

    print(
        f"Arquivos encontrados: "
        f"{len(meta_files):,}"
    )

    writer = None

    try:

        for meta_path in tqdm(
            meta_files,
            desc="Consolidando metadata",
            unit="arquivo",
        ):

            table = pq.read_table(
                meta_path
            )

            if writer is None:

                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(
                table
            )

            del table

    finally:

        if writer is not None:
            writer.close()

    print()
    print(
        f"Metadata final:\n"
        f"  {os.path.abspath(output_path)}"
    )


# ============================================================
# LIMPEZA DE TEMPORÁRIOS
# ============================================================

def cleanup_temp_files(
    output_dir,
):

    removed = 0

    # --------------------------------------------------------
    # NPZ temporários
    # --------------------------------------------------------

    for root, dirs, files in os.walk(
        output_dir
    ):

        for filename in files:

            if (
                filename.endswith(
                    ".npz.tmp"
                )
                or filename.endswith(
                    ".parquet.tmp"
                )
                or filename.endswith(
                    ".json.tmp"
                )
            ):

                path = os.path.join(
                    root,
                    filename,
                )

                try:

                    os.remove(
                        path
                    )

                    removed += 1

                except OSError:
                    pass

    if removed:

        print(
            f"Temporários removidos: "
            f"{removed}"
        )


# ============================================================
# VERIFICAÇÃO DOS CHUNKS
# ============================================================

def verify_chunks(
    output_dir,
    metadata_parts_dir,
):

    print()
    print(
        "Verificando chunks..."
    )

    problems = []

    for filename in os.listdir(
        metadata_parts_dir
    ):

        if not filename.endswith(
            "_meta.parquet"
        ):

            continue

        base = filename[
            :-len("_meta.parquet")
        ]

        npz_path = os.path.join(
            output_dir,
            base + ".npz",
        )

        meta_path = os.path.join(
            metadata_parts_dir,
            filename,
        )

        if not os.path.exists(
            npz_path
        ):

            problems.append(
                (
                    filename,
                    "NPZ ausente",
                )
            )

        if not os.path.exists(
            meta_path
        ):

            problems.append(
                (
                    filename,
                    "metadata ausente",
                )
            )

    if problems:

        print(
            "\nPROBLEMAS ENCONTRADOS:"
        )

        for filename, reason in problems:

            print(
                f"  {filename}: {reason}"
            )

        raise RuntimeError(
            "Existem chunks incompletos."
        )

    print(
        "Todos os chunks estão completos."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "TOKENIZAÇÃO STREAMING"
    )
    print(
        "SEM SPLIT_INDICES EM RAM"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Diretórios
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "metadata_parts",
        ),
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Remove temporários de execução anterior
    # --------------------------------------------------------

    cleanup_temp_files(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Verifica configuração
    # --------------------------------------------------------

    verify_processing_config()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nCarregando tokenizer..."
    )

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    pad_id = tokenizer.token_to_id(
        "<PAD>"
    )

    if pad_id is None:

        raise RuntimeError(
            "O tokenizer não possui "
            "o token <PAD>."
        )

    print(
        f"Tokenizer: "
        f"{TOKENIZER_PATH}"
    )

    print(
        f"PAD ID: {pad_id}"
    )

    print(
        f"MAX_SRC_LENGTH: "
        f"{MAX_SRC_LENGTH}"
    )

    print(
        f"MAX_TGT_LENGTH: "
        f"{MAX_TGT_LENGTH}"
    )

    print(
        f"NPZ_CHUNK_SIZE: "
        f"{NPZ_CHUNK_SIZE:,}".replace(
            ",",
            ".",
        )
    )

    print(
        f"PARQUET_BATCH_SIZE: "
        f"{PARQUET_BATCH_SIZE:,}".replace(
            ",",
            ".",
        )
    )

    print(
        f"MAX_WORKERS: "
        f"{MAX_WORKERS}"
    )

    # --------------------------------------------------------
    # Verifica Parquet
    # --------------------------------------------------------

    parquet_file = pq.ParquetFile(
        SPLIT_PARQUET_PATH
    )

    parquet_rows = (
        parquet_file.metadata.num_rows
    )

    print()
    print(
        f"Linhas no Parquet: "
        f"{parquet_rows:,}".replace(
            ",",
            ".",
        )
    )

    # --------------------------------------------------------
    # Processa
    # --------------------------------------------------------

    process_dataset(
        tsv_path=TSV_PATH,
        parquet_path=SPLIT_PARQUET_PATH,
        output_dir=OUTPUT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        pad_id=pad_id,
        parquet_rows=parquet_rows
    )

    # --------------------------------------------------------
    # Verifica chunks
    # --------------------------------------------------------

    metadata_parts_dir = os.path.join(
        OUTPUT_DIR,
        "metadata_parts",
    )

    verify_chunks(
        OUTPUT_DIR,
        metadata_parts_dir,
    )

    # --------------------------------------------------------
    # Consolida
    # --------------------------------------------------------

    consolidate_metadata(
        metadata_parts_dir,
        OUTPUT_SPLIT_PARQUET,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

    print()
    print("NPZs:")
    print(
        os.path.abspath(
            OUTPUT_DIR
        )
    )

    print()
    print("Metadata:")
    print(
        os.path.abspath(
            OUTPUT_SPLIT_PARQUET
        )
    )

    print()
    print("Checkpoint:")
    print(
        os.path.abspath(
            CHECKPOINT_PATH
        )
    )