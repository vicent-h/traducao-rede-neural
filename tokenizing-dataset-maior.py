import os
import gc
import io
import json
import time
import struct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

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

TOKENIZER_PATH = "artifacts/tokenizer_en_pt_es_60000.json"


# ============================================================
# PARÂMETROS
# ============================================================

MAX_SRC_LENGTH = 512
MAX_TGT_LENGTH = 512

# Quantos exemplos cabem em cada shard.
#
# Um shard NÃO é um exemplo: é um container com vários
# exemplos individuais, gravados sequencialmente.
#
# 100.000 exemplos x ~2 KB/registro ~= ~200 MB/shard
SHARD_SIZE = 100_000

# Quantas linhas do Parquet/TSV são lidas e tokenizadas por batch.
PARQUET_BATCH_SIZE = 10_000

# Quantos batches podem ficar simultaneamente nos workers.
MAX_PENDING = 16

# Número de processos de tokenização.
MAX_WORKERS = 16

# Faz flush do arquivo depois de cada exemplo.
# Isso deixa os dados persistidos no processo Python sem esperar
# o shard terminar.
#
# fsync é opcional porque fazer fsync em cada exemplo pode reduzir
# drasticamente a velocidade.
FLUSH_EACH_EXAMPLE = True
FSYNC_EACH_EXAMPLE = False

# Quantos registros são usados por vez ao consolidar JSONL -> Parquet.
METADATA_CONSOLIDATE_BATCH = 50_000


# ============================================================
# CHECKPOINT / CONFIG
# ============================================================

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "checkpoint.json",
)

CONFIG_PATH = os.path.join(
    OUTPUT_DIR,
    "processing_config.json",
)

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)


# ============================================================
# UTILITÁRIOS
# ============================================================

def atomic_write_json(path, data):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, path)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return {
            "tsv_rows_consumed": 0,
            "processed_examples": 0,
        }

    print(f"\nCheckpoint encontrado: {CHECKPOINT_PATH}")

    with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    tsv_rows_consumed,
    processed_examples,
):
    atomic_write_json(
        CHECKPOINT_PATH,
        {
            "tsv_rows_consumed": int(tsv_rows_consumed),
            "processed_examples": int(processed_examples),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )


def get_processing_config():
    return {
        "tsv_path": os.path.abspath(TSV_PATH),
        "split_parquet_path": os.path.abspath(SPLIT_PARQUET_PATH),
        "tokenizer_path": os.path.abspath(TOKENIZER_PATH),
        "max_src_length": MAX_SRC_LENGTH,
        "max_tgt_length": MAX_TGT_LENGTH,
        "shard_size": SHARD_SIZE,
        "parquet_batch_size": PARQUET_BATCH_SIZE,
        "max_pending": MAX_PENDING,
        "max_workers": MAX_WORKERS,
        "format": "fixed_record_binary_shards_v1",
    }


def verify_processing_config():
    current = get_processing_config()

    if not os.path.exists(CONFIG_PATH):
        atomic_write_json(CONFIG_PATH, current)
        return

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        old = json.load(f)

    if old != current:
        print("\nConfiguração anterior:")
        print(json.dumps(old, indent=2, ensure_ascii=False))

        print("\nConfiguração atual:")
        print(json.dumps(current, indent=2, ensure_ascii=False))

        raise RuntimeError(
            "\nA configuração mudou desde o último processamento.\n"
            "Para evitar misturar formatos ou duplicar dados, "
            "o processamento foi interrompido.\n\n"
            "Se quiser realmente começar do zero, remova o "
            "checkpoint/configuração e os shards gerados."
        )


def cleanup_temp_files(output_dir):
    removed = 0

    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith(
                (".tmp", ".bin.tmp", ".jsonl.tmp")
            ):
                path = os.path.join(root, filename)

                try:
                    os.remove(path)
                    removed += 1
                except OSError:
                    pass

    if removed:
        print(f"Temporários removidos: {removed}")


# ============================================================
# PADDING
# ============================================================

def pad_tokens(tokens, max_length, pad_id):
    tokens = tokens[:max_length]
    length = len(tokens)

    if length < max_length:
        tokens = tokens + [pad_id] * (max_length - length)

    return tokens, length


# ============================================================
# FORMATO DOS REGISTROS
# ============================================================
#
# Cada exemplo ocupa sempre exatamente o mesmo número de bytes:
#
#   src_tokens   -> MAX_SRC_LENGTH int32
#   tgt_tokens   -> MAX_TGT_LENGTH int32
#   src_length   -> int32
#   tgt_length   -> int32
#
# Isso permite acesso direto:
#
#   offset = example_index * RECORD_SIZE
#
# Sem precisar abrir um arquivo separado para cada exemplo.
# ============================================================

RECORD_SIZE = (
    MAX_SRC_LENGTH * np.dtype(np.int32).itemsize
    + MAX_TGT_LENGTH * np.dtype(np.int32).itemsize
    + 2 * np.dtype(np.int32).itemsize
)


def encode_record(
    src_tokens,
    tgt_tokens,
    src_length,
    tgt_length,
):
    src_array = np.asarray(
        src_tokens,
        dtype=np.int32,
    )

    tgt_array = np.asarray(
        tgt_tokens,
        dtype=np.int32,
    )

    if src_array.shape != (MAX_SRC_LENGTH,):
        raise RuntimeError(
            f"src_tokens possui shape inesperado: {src_array.shape}"
        )

    if tgt_array.shape != (MAX_TGT_LENGTH,):
        raise RuntimeError(
            f"tgt_tokens possui shape inesperado: {tgt_array.shape}"
        )

    header = struct.pack(
        "<ii",
        int(src_length),
        int(tgt_length),
    )

    return (
        src_array.tobytes(order="C")
        + tgt_array.tobytes(order="C")
        + header
    )


# ============================================================
# SHARD WRITER
# ============================================================

class ShardWriter:
    """
    Grava exemplos individualmente em um container binário.

    Estrutura:

        tokenized/
            shards/
                train/
                    shard_000000.bin
                    shard_000000.jsonl
                    shard_000001.bin
                    shard_000001.jsonl
                    ...

    O .bin contém os tokens.
    O .jsonl contém uma linha de metadata para cada exemplo.

    Portanto:
        - tokenização continua em batch;
        - gravação é individual;
        - não existe um arquivo por exemplo;
        - poucos arquivos são criados;
        - um processo interrompido não exige refazer um shard inteiro.
    """

    def __init__(
        self,
        output_dir,
        split_name,
        shard_size,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.split_name = split_name
        self.shard_size = int(shard_size)

        self.shard_dir = os.path.join(
            self.output_dir,
            "shards",
            split_name,
        )

        os.makedirs(
            self.shard_dir,
            exist_ok=True,
        )

        self.shard_id = 0
        self.position = 0

        self.bin_file = None
        self.meta_file = None
        self.bin_path = None
        self.meta_path = None

        self._recover_existing()

    # --------------------------------------------------------
    # Descobre/reabre o último shard
    # --------------------------------------------------------

    def _recover_existing(self):
        shard_ids = []

        for filename in os.listdir(self.shard_dir):
            if not filename.startswith("shard_"):
                continue

            if not filename.endswith(".bin"):
                continue

            number = filename[
                len("shard_"):-len(".bin")
            ]

            try:
                shard_ids.append(int(number))
            except ValueError:
                pass

        if not shard_ids:
            self._open_shard(0, 0)
            return

        last_id = max(shard_ids)

        bin_path = os.path.join(
            self.shard_dir,
            f"shard_{last_id:06d}.bin",
        )

        meta_path = os.path.join(
            self.shard_dir,
            f"shard_{last_id:06d}.jsonl",
        )

        if not os.path.exists(meta_path):
            # Um .bin sem metadata correspondente não é considerado
            # completo. Mantemos somente shards anteriores.
            os.remove(bin_path)
            shard_ids.remove(last_id)

            if not shard_ids:
                self._open_shard(0, 0)
                return

            last_id = max(shard_ids)
            bin_path = os.path.join(
                self.shard_dir,
                f"shard_{last_id:06d}.bin",
            )
            meta_path = os.path.join(
                self.shard_dir,
                f"shard_{last_id:06d}.jsonl",
            )

        # ----------------------------------------------------
        # Recuperação do último shard
        #
        # O .bin pode ter recebido um registro e o processo
        # pode ter caído antes da metadata correspondente.
        #
        # Nesse caso, usamos o menor dos dois contadores.
        # ----------------------------------------------------

        bin_size = os.path.getsize(bin_path)
        bin_records = bin_size // RECORD_SIZE

        valid_bin_bytes = bin_records * RECORD_SIZE

        if bin_size != valid_bin_bytes:
            with open(bin_path, "r+b") as f:
                f.truncate(valid_bin_bytes)

        meta_count = self._count_jsonl_lines(meta_path)

        valid_count = min(
            bin_records,
            meta_count,
        )

        if bin_records != valid_count:
            with open(bin_path, "r+b") as f:
                f.truncate(valid_count * RECORD_SIZE)

        if meta_count != valid_count:
            self._truncate_jsonl_to_lines(
                meta_path,
                valid_count,
            )

        # Se o último shard já está cheio, começa outro.
        if valid_count >= self.shard_size:
            self._open_shard(
                last_id + 1,
                0,
            )
        else:
            self._open_shard(
                last_id,
                valid_count,
            )

    @staticmethod
    def _count_jsonl_lines(path):
        if not os.path.exists(path):
            return 0

        count = 0

        with open(
            path,
            "rb",
        ) as f:
            for line in f:
                if line.strip():
                    count += 1

        return count

    @staticmethod
    def _truncate_jsonl_to_lines(path, count):
        if count <= 0:
            with open(path, "wb"):
                pass
            return

        temp_path = path + ".tmp"

        written = 0

        with open(path, "rb") as src, open(
            temp_path,
            "wb",
        ) as dst:

            for line in src:
                if not line.strip():
                    continue

                if written >= count:
                    break

                dst.write(line)
                written += 1

        os.replace(temp_path, path)

    # --------------------------------------------------------
    # Abre shard
    # --------------------------------------------------------

    def _open_shard(
        self,
        shard_id,
        position,
    ):
        self.close()

        self.shard_id = int(shard_id)
        self.position = int(position)

        base = f"shard_{self.shard_id:06d}"

        self.bin_path = os.path.join(
            self.shard_dir,
            base + ".bin",
        )

        self.meta_path = os.path.join(
            self.shard_dir,
            base + ".jsonl",
        )

        self.bin_file = open(
            self.bin_path,
            "ab",
            buffering=1024 * 1024,
        )

        self.meta_file = open(
            self.meta_path,
            "a",
            encoding="utf-8",
            buffering=1024 * 1024,
        )

    # --------------------------------------------------------
    # Add individual
    # --------------------------------------------------------

    def add(
        self,
        global_row,
        dataset_id,
        index,
        src_tokens,
        tgt_tokens,
        pad_id,
    ):
        if self.position >= self.shard_size:
            self._open_shard(
                self.shard_id + 1,
                0,
            )

        src_padded, src_length = pad_tokens(
            src_tokens,
            MAX_SRC_LENGTH,
            pad_id,
        )

        tgt_padded, tgt_length = pad_tokens(
            tgt_tokens,
            MAX_TGT_LENGTH,
            pad_id,
        )

        record = encode_record(
            src_padded,
            tgt_padded,
            src_length,
            tgt_length,
        )

        if len(record) != RECORD_SIZE:
            raise RuntimeError(
                f"Registro possui {len(record)} bytes, "
                f"mas deveria possuir {RECORD_SIZE}."
            )

        shard_index = self.position

        # ----------------------------------------------------
        # 1. Tokens
        # ----------------------------------------------------

        self.bin_file.write(record)

        if FLUSH_EACH_EXAMPLE:
            self.bin_file.flush()

            if FSYNC_EACH_EXAMPLE:
                os.fsync(
                    self.bin_file.fileno()
                )

        # ----------------------------------------------------
        # 2. Metadata
        # ----------------------------------------------------

        metadata = {
            "global_row": int(global_row),
            "dataset_id": str(dataset_id),
            "index": int(index),
            "split": self.split_name,
            "shard": int(self.shard_id),
            "shard_index": int(shard_index),
            "shard_path": os.path.abspath(
                self.bin_path
            ),
            "src_length": int(src_length),
            "tgt_length": int(tgt_length),
        }

        self.meta_file.write(
            json.dumps(
                metadata,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

        if FLUSH_EACH_EXAMPLE:
            self.meta_file.flush()

            if FSYNC_EACH_EXAMPLE:
                os.fsync(
                    self.meta_file.fileno()
                )

        self.position += 1

    # --------------------------------------------------------
    # Última linha global salva
    # --------------------------------------------------------

    def get_last_global_row(self):
        if self.position <= 0:
            return None

        try:
            with open(
                self.meta_path,
                "rb",
            ) as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()

                if size == 0:
                    return None

                # Lê somente o final do arquivo.
                read_size = min(size, 1024 * 1024)

                f.seek(-read_size, os.SEEK_END)
                data = f.read()

            lines = data.splitlines()

            for line in reversed(lines):
                if line.strip():
                    row = json.loads(
                        line.decode("utf-8")
                    )
                    return int(row["global_row"])

        except Exception:
            return None

        return None

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    def close(self):
        if self.bin_file is not None:
            try:
                self.bin_file.flush()
            except Exception:
                pass

            try:
                self.bin_file.close()
            except Exception:
                pass

            self.bin_file = None

        if self.meta_file is not None:
            try:
                self.meta_file.flush()
            except Exception:
                pass

            try:
                self.meta_file.close()
            except Exception:
                pass

            self.meta_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================
# RECUPERAÇÃO DO PROGRESSO REAL DOS SHARDS
# ============================================================

def get_existing_progress(output_dir):
    """
    Descobre o maior global_row realmente persistido.

    Isso é importante porque o checkpoint pode ter sido salvo
    alguns instantes antes/depois de uma queda. Os próprios
    shards são tratados como fonte de verdade para os dados.
    """

    last_rows = []

    shards_root = os.path.join(
        output_dir,
        "shards",
    )

    if not os.path.exists(shards_root):
        return -1, 0

    for split_name in ("train", "val", "test"):
        split_dir = os.path.join(
            shards_root,
            split_name,
        )

        if not os.path.exists(split_dir):
            continue

        shard_files = sorted(
            filename
            for filename in os.listdir(split_dir)
            if filename.endswith(".jsonl")
        )

        if not shard_files:
            continue

        last_meta = os.path.join(
            split_dir,
            shard_files[-1],
        )

        with open(
            last_meta,
            "rb",
        ) as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()

            if size == 0:
                continue

            read_size = min(
                size,
                1024 * 1024,
            )

            f.seek(-read_size, os.SEEK_END)
            data = f.read()

        for line in reversed(
            data.splitlines()
        ):
            if line.strip():
                obj = json.loads(
                    line.decode("utf-8")
                )
                last_rows.append(
                    int(obj["global_row"])
                )
                break

    if not last_rows:
        return -1, 0

    return (
        max(last_rows),
        len(last_rows),
    )


# ============================================================
# WORKER
# ============================================================

_WORKER_TOKENIZER = None


def init_worker(tokenizer_path):
    global _WORKER_TOKENIZER

    _WORKER_TOKENIZER = Tokenizer.from_file(
        tokenizer_path
    )


def tokenize_batch(rows):
    """
    Tokenização continua sendo feita em batch.

    Cada row contém:
        (
            global_row,
            split,
            dataset_id,
            index,
            text1,
            text2,
        )
    """

    results = []

    for (
        global_row,
        split_name,
        dataset_id,
        idx,
        text1,
        text2,
    ) in rows:

        if dataset_id.endswith("_en_pt"):
            text2 = f"<2pt> {text2}"

        elif dataset_id.endswith("_en_es"):
            text2 = f"<2es> {text2}"

        try:
            src_encoded = _WORKER_TOKENIZER.encode(
                text1
            )

            tgt_encoded = _WORKER_TOKENIZER.encode(
                text2
            )

        except Exception as exc:
            # Não silencie erros de tokenização.
            # Um exemplo perdido quebraria o alinhamento do
            # checkpoint.
            raise RuntimeError(
                f"Falha ao tokenizar global_row={global_row}, "
                f"dataset_id={dataset_id}, index={idx}: {exc}"
            ) from exc

        results.append(
            (
                global_row,
                split_name,
                dataset_id,
                idx,
                src_encoded.ids,
                tgt_encoded.ids,
            )
        )

    return results


# ============================================================
# PARSE TSV
# ============================================================

def parse_tsv_line(line):
    line = line.rstrip("\n")

    try:
        meta_part, texts_part = line.split(
            "<METADATA>",
            1,
        )

        dataset_id, idx_str = meta_part.split(
            "<SEP>",
            1,
        )

        text1, text2 = texts_part.split(
            "<SEP>",
            1,
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
# PARQUET
# ============================================================

def load_parquet_dataframe(parquet_path):
    print("\nCarregando Parquet inteiro em memória...")

    df = pd.read_parquet(
        parquet_path,
        columns=[
            "dataset_id",
            "index",
            "split",
        ],
    )

    print(
        f"Parquet carregado: "
        f"{len(df):,}".replace(",", ".")
        + " linhas"
    )

    return df


# ============================================================
# COMMIT ORDENADO
# ============================================================

def commit_results(
    results,
    writers,
    pad_id,
):
    """
    Grava cada exemplo individualmente.

    A ordem global é mantida pelo process_dataset().
    """

    for (
        global_row,
        split_name,
        dataset_id,
        idx,
        src_tokens,
        tgt_tokens,
    ) in results:

        writers[split_name].add(
            global_row=global_row,
            dataset_id=dataset_id,
            index=idx,
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            pad_id=pad_id,
        )


# ============================================================
# PROCESSAMENTO
# ============================================================

def process_dataset(
    tsv_path,
    parquet_df,
    output_dir,
    tokenizer_path,
    pad_id,
):
    total_rows = len(parquet_df)

    checkpoint = load_checkpoint()

    checkpoint_row = int(
        checkpoint.get(
            "tsv_rows_consumed",
            0,
        )
    )

    checkpoint_processed = int(
        checkpoint.get(
            "processed_examples",
            0,
        )
    )

    # --------------------------------------------------------
    # Cria/reabre writers.
    #
    # O próprio conteúdo dos shards é usado para recuperar
    # o progresso real.
    # --------------------------------------------------------

    writers = {
        "train": ShardWriter(
            output_dir,
            "train",
            SHARD_SIZE,
        ),
        "val": ShardWriter(
            output_dir,
            "val",
            SHARD_SIZE,
        ),
        "test": ShardWriter(
            output_dir,
            "test",
            SHARD_SIZE,
        ),
    }

    shard_last_row, _ = get_existing_progress(
        output_dir
    )

    recovered_row = shard_last_row + 1

    # O maior valor é usado para evitar voltar para trás.
    start_row = max(
        checkpoint_row,
        recovered_row,
    )

    if start_row > total_rows:
        raise RuntimeError(
            f"Checkpoint/shards indicam {start_row:,} linhas, "
            f"mas o Parquet possui {total_rows:,}."
        )

    if recovered_row != checkpoint_row:
        print(
            "\nRecuperação pelos shards:"
        )
        print(
            f"  checkpoint: {checkpoint_row:,}"
        )
        print(
            f"  shards:     {recovered_row:,}"
        )
        print(
            f"  continuará em: {start_row:,}"
        )

    # Se não houve dados anteriores, processed_examples vem do zero.
    # Em uma retomada, o contador do checkpoint continua sendo útil
    # apenas como informação; o progresso real é determinado pelos
    # shards.
    processed_examples = max(
        checkpoint_processed,
        0,
    )

    print()
    print("=" * 70)
    print("PROCESSAMENTO")
    print("=" * 70)

    print(
        f"Total de linhas:       {total_rows:,}".replace(",", ".")
    )
    print(
        f"Já persistidas:        {start_row:,}".replace(",", ".")
    )
    print(
        f"Restantes:             {total_rows - start_row:,}".replace(",", ".")
    )
    print(
        f"Batch de tokenização:  {PARQUET_BATCH_SIZE:,}".replace(",", ".")
    )
    print(
        f"Exemplos por shard:    {SHARD_SIZE:,}".replace(",", ".")
    )
    print(
        f"Tamanho do registro:   {RECORD_SIZE:,} bytes".replace(",", ".")
    )
    print(
        f"~tamanho máximo shard: "
        f"{RECORD_SIZE * SHARD_SIZE / (1024 ** 2):,.1f} MiB"
    )
    print(
        f"Workers:               {min(MAX_WORKERS, os.cpu_count() or 1)}"
    )

    # --------------------------------------------------------
    # TSV em streaming.
    # --------------------------------------------------------

    with open(
        tsv_path,
        "r",
        encoding="utf-8",
    ) as tsv:

        # Cabeçalho
        next(tsv)

        # ----------------------------------------------------
        # Pula até a posição recuperada.
        # ----------------------------------------------------

        if start_row > 0:
            print("\nRecuperando posição do TSV...")

            with tqdm(
                total=start_row,
                initial=0,
                desc="Recuperando TSV",
                unit="linha",
                dynamic_ncols=True,
            ) as pbar_recovery:

                for _ in range(start_row):
                    line = tsv.readline()

                    if not line:
                        raise RuntimeError(
                            "O TSV terminou antes do ponto "
                            "recuperado."
                        )

                    pbar_recovery.update(1)

        workers = min(
            MAX_WORKERS,
            os.cpu_count() or 1,
        )

        # fork é adequado ao Linux usado no ambiente do usuário.
        mp_context = mp.get_context("fork")

        pending = {}
        completed = {}

        current_row = start_row
        next_commit_row = start_row

        pbar = tqdm(
            total=total_rows,
            initial=start_row,
            desc="Processando",
            unit="linha",
            dynamic_ncols=True,
        )

        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=init_worker,
                initargs=(tokenizer_path,),
            ) as executor:

                # ------------------------------------------------
                # Alimenta os workers e limita MAX_PENDING.
                # ------------------------------------------------

                while (
                    current_row < total_rows
                    or pending
                    or completed
                ):

                    while (
                        current_row < total_rows
                        and len(pending) < MAX_PENDING
                    ):

                        batch_start = current_row
                        batch_end = min(
                            batch_start + PARQUET_BATCH_SIZE,
                            total_rows,
                        )

                        parquet_chunk = parquet_df.iloc[
                            batch_start:batch_end
                        ]

                        batch_rows = []

                        for local_offset, row in enumerate(
                            parquet_chunk.itertuples(
                                index=False
                            )
                        ):
                            global_row = (
                                batch_start
                                + local_offset
                            )

                            line = tsv.readline()

                            if not line:
                                raise RuntimeError(
                                    "TSV terminou antes do Parquet."
                                )

                            parsed = parse_tsv_line(line)

                            if parsed is None:
                                raise RuntimeError(
                                    "Linha malformada no TSV "
                                    f"na posição {global_row}."
                                )

                            (
                                dataset_id_tsv,
                                index_tsv,
                                text1,
                                text2,
                            ) = parsed

                            # Verificação crítica.
                            if (
                                dataset_id_tsv
                                != row.dataset_id
                                or int(index_tsv)
                                != int(row.index)
                            ):
                                raise RuntimeError(
                                    "\n\nERRO DE ORDEM!\n"
                                    "Parquet e TSV não estão alinhados.\n\n"
                                    f"Posição: {global_row}\n\n"
                                    f"TSV:\n"
                                    f"  dataset_id = {dataset_id_tsv}\n"
                                    f"  index      = {index_tsv}\n\n"
                                    f"Parquet:\n"
                                    f"  dataset_id = {row.dataset_id}\n"
                                    f"  index      = {row.index}\n"
                                )

                            batch_rows.append(
                                (
                                    global_row,
                                    row.split,
                                    dataset_id_tsv,
                                    index_tsv,
                                    text1,
                                    text2,
                                )
                            )

                        if not batch_rows:
                            break

                        future = executor.submit(
                            tokenize_batch,
                            batch_rows,
                        )

                        pending[future] = (
                            batch_start,
                            batch_end,
                        )

                        current_row = batch_end

                        del parquet_chunk
                        del batch_rows

                    # ------------------------------------------------
                    # Espera pelo menos um worker.
                    # ------------------------------------------------

                    if pending:
                        done, _ = wait(
                            list(pending.keys()),
                            return_when=FIRST_COMPLETED,
                        )

                        for future in done:
                            batch_start, batch_end = pending.pop(
                                future
                            )

                            results = future.result()

                            expected = (
                                batch_end - batch_start
                            )

                            if len(results) != expected:
                                raise RuntimeError(
                                    f"O batch {batch_start}:{batch_end} "
                                    f"retornou {len(results)} resultados, "
                                    f"mas esperávamos {expected}."
                                )

                            completed[batch_start] = (
                                batch_end,
                                results,
                            )

                    # ------------------------------------------------
                    # Commit estritamente em ordem.
                    #
                    # Isso permite que o checkpoint continue
                    # representando uma posição contínua no TSV.
                    # ------------------------------------------------

                    while next_commit_row in completed:
                        batch_end, results = completed.pop(
                            next_commit_row
                        )

                        commit_results(
                            results,
                            writers,
                            pad_id,
                        )

                        count = batch_end - next_commit_row

                        processed_examples += count

                        pbar.update(count)

                        next_commit_row = batch_end

                        # O checkpoint só avança depois que todos os
                        # exemplos do batch foram efetivamente gravados.
                        save_checkpoint(
                            tsv_rows_consumed=next_commit_row,
                            processed_examples=processed_examples,
                        )

                        del results

                # ----------------------------------------------------
                # Sanidade final
                # ----------------------------------------------------

                if next_commit_row != total_rows:
                    raise RuntimeError(
                        f"Processamento terminou em {next_commit_row:,}, "
                        f"mas o dataset possui {total_rows:,} linhas."
                    )

        finally:
            pbar.close()

    for writer in writers.values():
        writer.close()

    save_checkpoint(
        tsv_rows_consumed=total_rows,
        processed_examples=processed_examples,
    )

    print()
    print("=" * 70)
    print("TOKENIZAÇÃO FINALIZADA")
    print("=" * 70)
    print(
        f"Linhas consumidas: {total_rows:,}".replace(",", ".")
    )
    print(
        f"Exemplos processados: {processed_examples:,}".replace(",", ".")
    )


# ============================================================
# CONSOLIDA METADATA
# ============================================================

def iter_metadata_jsonl(
    metadata_path,
    batch_size,
):
    batch = []

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            if not line.strip():
                continue

            batch.append(
                json.loads(line)
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def consolidate_metadata(
    output_dir,
    output_path,
):
    """
    Converte os JSONL individuais dos shards para um único
    Parquet final.

    A consolidação ocorre somente depois da tokenização.
    """

    print()
    print("=" * 70)
    print("CONSOLIDANDO METADADOS")
    print("=" * 70)

    shards_root = os.path.join(
        output_dir,
        "shards",
    )

    meta_files = []

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_dir = os.path.join(
            shards_root,
            split_name,
        )

        if not os.path.exists(split_dir):
            continue

        for filename in os.listdir(split_dir):
            if filename.endswith(".jsonl"):
                meta_files.append(
                    os.path.join(
                        split_dir,
                        filename,
                    )
                )

    meta_files.sort()

    if not meta_files:
        raise RuntimeError(
            "Nenhum metadata JSONL encontrado."
        )

    print(
        f"Shards de metadata: {len(meta_files):,}".replace(
            ",", "."
        )
    )

    writer = None

    schema = pa.schema(
        [
            ("global_row", pa.int64()),
            ("dataset_id", pa.string()),
            ("index", pa.int64()),
            ("split", pa.string()),
            ("shard", pa.int64()),
            ("shard_index", pa.int64()),
            ("shard_path", pa.string()),
            ("src_length", pa.int32()),
            ("tgt_length", pa.int32()),
        ]
    )

    try:
        for meta_path in tqdm(
            meta_files,
            desc="Metadata",
            unit="shard",
            dynamic_ncols=True,
        ):
            for batch in iter_metadata_jsonl(
                meta_path,
                METADATA_CONSOLIDATE_BATCH,
            ):
                table = pa.Table.from_pylist(
                    batch,
                    schema=schema,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        schema,
                        compression="zstd",
                    )

                writer.write_table(table)

                del table
                del batch

    finally:
        if writer is not None:
            writer.close()

    print(
        f"\nMetadata final:\n  "
        f"{os.path.abspath(output_path)}"
    )


# ============================================================
# VERIFICAÇÃO
# ============================================================

def verify_shards(output_dir):
    """
    Verificação estrutural rápida.

    Para cada shard:
      - .bin precisa existir;
      - .jsonl precisa existir;
      - tamanho do .bin precisa ser múltiplo de RECORD_SIZE;
      - quantidade de linhas do JSONL precisa bater com a
        quantidade de registros do .bin.

    Essa etapa percorre as linhas dos JSONL, portanto pode levar
    algum tempo em um dataset de dezenas de milhões de exemplos.
    """

    print()
    print("=" * 70)
    print("VERIFICANDO SHARDS")
    print("=" * 70)

    shards_root = os.path.join(
        output_dir,
        "shards",
    )

    total_records = 0
    problems = []

    for split_name in (
        "train",
        "val",
        "test",
    ):
        split_dir = os.path.join(
            shards_root,
            split_name,
        )

        if not os.path.exists(split_dir):
            continue

        bin_files = sorted(
            filename
            for filename in os.listdir(split_dir)
            if filename.endswith(".bin")
        )

        for bin_filename in tqdm(
            bin_files,
            desc=f"Verificando {split_name}",
            unit="shard",
            dynamic_ncols=True,
        ):
            bin_path = os.path.join(
                split_dir,
                bin_filename,
            )

            meta_path = os.path.splitext(
                bin_path
            )[0] + ".jsonl"

            if not os.path.exists(meta_path):
                problems.append(
                    (bin_path, "JSONL ausente")
                )
                continue

            bin_size = os.path.getsize(
                bin_path
            )

            if bin_size % RECORD_SIZE != 0:
                problems.append(
                    (
                        bin_path,
                        "tamanho não é múltiplo de RECORD_SIZE",
                    )
                )
                continue

            bin_count = (
                bin_size // RECORD_SIZE
            )

            meta_count = ShardWriter._count_jsonl_lines(
                meta_path
            )

            if bin_count != meta_count:
                problems.append(
                    (
                        bin_path,
                        f"bin={bin_count}, metadata={meta_count}",
                    )
                )
            else:
                total_records += bin_count

    if problems:
        print("\nPROBLEMAS ENCONTRADOS:")

        for path, reason in problems[:50]:
            print(
                f"  {path}: {reason}"
            )

        if len(problems) > 50:
            print(
                f"  ... e mais {len(problems) - 50} problemas."
            )

        raise RuntimeError(
            "Existem shards inconsistentes."
        )

    print(
        f"\nTodos os shards estão consistentes."
    )
    print(
        f"Total de registros: "
        f"{total_records:,}".replace(",", ".")
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TOKENIZAÇÃO COM SHARDS")
    print("PARQUET EM RAM + TSV EM STREAMING")
    print("TOKENIZAÇÃO EM BATCH + SALVAMENTO INDIVIDUAL")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            "shards",
        ),
        exist_ok=True,
    )

    cleanup_temp_files(
        OUTPUT_DIR
    )

    verify_processing_config()

    print("\nCarregando tokenizer...")

    tokenizer = Tokenizer.from_file(
        TOKENIZER_PATH
    )

    pad_id = tokenizer.token_to_id(
        "<PAD>"
    )

    if pad_id is None:
        raise RuntimeError(
            "O tokenizer não possui o token <PAD>."
        )

    print(
        f"Tokenizer: {TOKENIZER_PATH}"
    )
    print(
        f"PAD ID: {pad_id}"
    )
    print(
        f"MAX_SRC_LENGTH: {MAX_SRC_LENGTH}"
    )
    print(
        f"MAX_TGT_LENGTH: {MAX_TGT_LENGTH}"
    )
    print(
        f"SHARD_SIZE: {SHARD_SIZE:,}".replace(",", ".")
    )
    print(
        f"PARQUET_BATCH_SIZE: "
        f"{PARQUET_BATCH_SIZE:,}".replace(",", ".")
    )
    print(
        f"MAX_PENDING: {MAX_PENDING}"
    )
    print(
        f"MAX_WORKERS: {MAX_WORKERS}"
    )
    print(
        f"RECORD_SIZE: {RECORD_SIZE:,} bytes".replace(
            ",", "."
        )
    )

    parquet_df = load_parquet_dataframe(
        SPLIT_PARQUET_PATH
    )

    parquet_rows = len(parquet_df)

    print(
        f"Linhas no dataset: "
        f"{parquet_rows:,}".replace(",", ".")
    )

    process_dataset(
        tsv_path=TSV_PATH,
        parquet_df=parquet_df,
        output_dir=OUTPUT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        pad_id=pad_id,
    )

    del parquet_df
    del tokenizer
    gc.collect()

    # --------------------------------------------------------
    # Verificação final.
    # --------------------------------------------------------

    verify_shards(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Consolida metadata.
    # --------------------------------------------------------

    consolidate_metadata(
        OUTPUT_DIR,
        OUTPUT_SPLIT_PARQUET,
    )

    print()
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

    print("\nShards:")
    print(
        os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                "shards",
            )
        )
    )

    print("\nMetadata:")
    print(
        os.path.abspath(
            OUTPUT_SPLIT_PARQUET
        )
    )

    print("\nCheckpoint:")
    print(
        os.path.abspath(
            CHECKPOINT_PATH
        )
    )
