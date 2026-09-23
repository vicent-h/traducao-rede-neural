#!/usr/bin/env python3
"""
Converte os shards binários FIXOS atuais para shards binários VARIÁVEIS.

IMPORTANTE:
O metadata atual já possui:
    global_row
    dataset_id
    index
    split
    shard
    shard_index
    shard_path
    src_length
    tgt_length

Portanto NÃO existe npz_path/npz_index e não devemos procurar essas colunas.

Formato esperado do shard antigo (gerado pelo script de shards anterior):

    int32 src_tokens[256]
    int32 tgt_tokens[256]
    int32 src_length
    int32 tgt_length

Formato do novo shard:

    uint32 src_length
    uint32 tgt_length
    int32  src_tokens[src_length]
    int32  tgt_tokens[tgt_length]

Novo metadata:

    global_row
    dataset_id
    index
    split
    shard
    shard_index
    shard_path
    offset
    src_length
    tgt_length

O script NÃO retokeniza nada.
Ele apenas remove o padding físico dos shards antigos.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FIXED_MAX_SRC = 512
FIXED_MAX_TGT = 512
TOKEN_DTYPE = np.dtype("<i4")
HEADER_DTYPE = np.dtype("<u4")

OLD_RECORD_BYTES = (
    (FIXED_MAX_SRC + FIXED_MAX_TGT) * TOKEN_DTYPE.itemsize
    + 2 * TOKEN_DTYPE.itemsize
)

NEW_HEADER_BYTES = 8


class VariableShardWriter:
    def __init__(self, output_dir, split, shard_size):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.split = str(split)
        self.shard_size = int(shard_size)

        self.shard_id = self._next_shard_id()

        self.f = None
        self.path = None
        self.count = 0
        self.offset = 0

    def _next_shard_id(self):
        ids = []

        for p in self.output_dir.glob(f"{self.split}_*.bin"):
            try:
                ids.append(int(p.stem.rsplit("_", 1)[1]))
            except (ValueError, IndexError):
                pass

        return max(ids) + 1 if ids else 0

    def _open(self):
        self.path = (
            self.output_dir
            / f"{self.split}_{self.shard_id:05d}.bin"
        )
        self.f = open(self.path, "wb")
        self.count = 0
        self.offset = 0

    def add(self, src, tgt):
        if self.f is None:
            self._open()

        src = np.asarray(src, dtype=TOKEN_DTYPE)
        tgt = np.asarray(tgt, dtype=TOKEN_DTYPE)

        record_offset = self.offset

        self.f.write(
            np.asarray(
                [len(src), len(tgt)],
                dtype=HEADER_DTYPE,
            ).tobytes()
        )

        self.f.write(src.tobytes(order="C"))
        self.f.write(tgt.tobytes(order="C"))

        self.offset += (
            NEW_HEADER_BYTES
            + (len(src) + len(tgt)) * TOKEN_DTYPE.itemsize
        )

        self.count += 1

        return record_offset

    def close_shard(self):
        if self.f is None:
            return

        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        self.f = None

        print(
            f"[{self.split}] shard {self.shard_id:05d}: "
            f"{self.count:,} exemplos | "
            f"{self.offset / 1024**2:.2f} MB"
        )

        self.shard_id += 1
        self.count = 0
        self.offset = 0
        self.path = None

    def close(self):
        self.close_shard()


def read_old_record(f, shard_index):
    offset = int(shard_index) * OLD_RECORD_BYTES

    f.seek(offset)

    raw = f.read(OLD_RECORD_BYTES)

    if len(raw) != OLD_RECORD_BYTES:
        raise RuntimeError(
            f"Registro incompleto: shard_index={shard_index}, "
            f"offset={offset}, bytes={len(raw)}, "
            f"esperado={OLD_RECORD_BYTES}"
        )

    values = np.frombuffer(
        raw,
        dtype=TOKEN_DTYPE,
    )

    src = values[:FIXED_MAX_SRC]
    tgt = values[
        FIXED_MAX_SRC:
        FIXED_MAX_SRC + FIXED_MAX_TGT
    ]

    src_length = int(values[-2])
    tgt_length = int(values[-1])

    if not 0 <= src_length <= FIXED_MAX_SRC:
        raise RuntimeError(
            f"src_length inválido: {src_length}"
        )

    if not 0 <= tgt_length <= FIXED_MAX_TGT:
        raise RuntimeError(
            f"tgt_length inválido: {tgt_length}"
        )

    return (
        src[:src_length].copy(),
        tgt[:tgt_length].copy(),
    )


def convert_split(metadata, split, output_dir, shard_size):
    split_df = metadata[
        metadata["split"].astype(str).str.lower()
        == split.lower()
    ].copy()

    if split_df.empty:
        print(f"[{split}] nenhum exemplo.")
        return

    required = {
        "global_row",
        "dataset_id",
        "index",
        "split",
        "shard",
        "shard_index",
        "shard_path",
        "src_length",
        "tgt_length",
    }

    missing = required - set(split_df.columns)

    if missing:
        raise RuntimeError(
            f"Metadata sem colunas obrigatórias para {split}: "
            f"{sorted(missing)}"
        )

    # A ordem precisa ser a mesma ordem física dos registros antigos.
    split_df = split_df.sort_values(
        ["shard", "shard_index"],
        kind="stable",
    )

    meta_rows = []

    writer = VariableShardWriter(
        output_dir=output_dir,
        split=split,
        shard_size=shard_size,
    )

    current_old_shard = None
    old_file = None

    try:
        for row in split_df.itertuples(index=False):
            old_path = os.path.abspath(str(row.shard_path))

            if old_path != current_old_shard:
                if old_file is not None:
                    old_file.close()

                if not os.path.isfile(old_path):
                    raise FileNotFoundError(old_path)

                print(f"[{split}] lendo shard antigo: {old_path}")

                old_file = open(old_path, "rb")
                current_old_shard = old_path

            src, tgt = read_old_record(
                old_file,
                row.shard_index,
            )

            # Confere os comprimentos do metadata.
            if len(src) != int(row.src_length):
                raise RuntimeError(
                    f"src_length inconsistente em "
                    f"{old_path}, shard_index={row.shard_index}: "
                    f"metadata={row.src_length}, "
                    f"arquivo={len(src)}"
                )

            if len(tgt) != int(row.tgt_length):
                raise RuntimeError(
                    f"tgt_length inconsistente em "
                    f"{old_path}, shard_index={row.shard_index}: "
                    f"metadata={row.tgt_length}, "
                    f"arquivo={len(tgt)}"
                )

            new_offset = writer.add(src, tgt)

            new_shard_path = str(
                writer.path.resolve()
            )

            meta_rows.append(
                {
                    "global_row": int(row.global_row),
                    "dataset_id": str(row.dataset_id),
                    "index": int(row.index),
                    "split": str(row.split),
                    "shard": int(writer.shard_id),
                    "shard_index": int(writer.count - 1),
                    "shard_path": new_shard_path,
                    "offset": int(new_offset),
                    "src_length": int(row.src_length),
                    "tgt_length": int(row.tgt_length),
                }
            )

            if writer.count >= writer.shard_size:
                writer.close_shard()

    finally:
        if old_file is not None:
            old_file.close()

        writer.close()

    return meta_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata Parquet atual, com shard_path/shard_index.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Diretório dos novos shards variáveis.",
    )

    parser.add_argument(
        "--final_metadata",
        default=None,
        help="Nome do novo metadata consolidado.",
    )

    parser.add_argument(
        "--shard_size",
        type=int,
        default=100_000,
    )

    args = parser.parse_args()

    metadata = pd.read_parquet(args.metadata)

    print("=" * 70)
    print("CONVERSÃO DE SHARDS FIXOS -> SHARDS VARIÁVEIS")
    print("=" * 70)
    print(f"Metadata: {args.metadata}")
    print(f"Saída:    {args.output_dir}")
    print(f"Registros/shard: {args.shard_size:,}")
    print()

    all_rows = []

    for split in ("train", "val", "test"):
        rows = convert_split(
            metadata=metadata,
            split=split,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
        )

        if rows:
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError(
            "Nenhum exemplo foi convertido."
        )

    result = pd.DataFrame(all_rows)

    result = result.sort_values(
        "global_row",
        kind="stable",
    ).reset_index(drop=True)

    output_dir = Path(args.output_dir)

    if args.final_metadata:
        metadata_name = args.final_metadata
    else:
        metadata_name = (
            Path(args.metadata).name
        )

    output_metadata = output_dir / metadata_name
    tmp_metadata = Path(
        str(output_metadata) + ".tmp"
    )

    table = pa.Table.from_pandas(
        result,
        preserve_index=False,
    )

    pq.write_table(
        table,
        tmp_metadata,
        compression="zstd",
    )

    os.replace(
        tmp_metadata,
        output_metadata,
    )

    print()
    print("=" * 70)
    print("CONVERSÃO CONCLUÍDA")
    print("=" * 70)
    print(f"Exemplos: {len(result):,}")
    print(f"Metadata: {output_metadata}")
    print()
    print("Novas colunas:")
    print(list(result.columns))


if __name__ == "__main__":
    main()
