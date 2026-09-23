from argparse import ArgumentParser
from datetime import datetime
import glob
import json
import logging
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.lstm_proj_diff import LSTM
from models.transformer import Transformer
from utils.earlystopper import EarlyStopper
from utils.scheduler_sampling import SigmoidSchedulerSampling
from utils.warmup import WarmupScheduler

logger = logging.getLogger(__name__)

MAX_TRAIN_OBSERVATIONS = None      # None = usa todo o train
MAX_VAL_OBSERVATIONS = 25_000      # Ex.: limita validação a 50 mil


def resolve_metadata_path(tokenized_dir: str, metadata_path: str | None = None) -> str:
    """
    Resolve o único arquivo de metadata consolidado.

    Por padrão usa:
        <tokenized_dir>/analise_textos_tokenized_split.parquet

    Também aceita --metadata_path explícito.
    """
    if metadata_path:
        path = os.path.abspath(metadata_path)
    else:
        path = os.path.join(
            os.path.abspath(tokenized_dir),
            "analise_textos_tokenized_split.parquet",
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Metadata consolidado não encontrado: {path}"
        )

    return path



class CurriculumLengthSampler(torch.utils.data.Sampler):
    """
    Curriculum por comprimento sem materializar índices em RAM.

    Como o treinamento agora usa num_workers=0, o estado do curriculum fica
    diretamente no processo principal.
    """

    def __init__(self, curriculum_levels):
        if not curriculum_levels:
            raise ValueError("curriculum_levels não pode estar vazio.")

        self.curriculum_levels = curriculum_levels
        self.step_count = 0
        self._level_index = 0

    @property
    def level_index(self):
        return self._level_index

    @property
    def level_actual(self):
        return self.curriculum_levels[self.level_index]

    def step(self):
        self.step_count += 1

        current_index = self.level_index
        current_level = self.curriculum_levels[current_index]

        if (
            self.step_count > current_level["max_step"]
            and current_index < len(self.curriculum_levels) - 1
        ):
            new_index = current_index + 1
            self._level_index = new_index

            new_level = self.curriculum_levels[new_index]

            logger.info(
                "Curriculum avançou para nível %d: max_len=%s, batch_size=%s, accum_steps=%s",
                new_index,
                new_level["max_len"],
                new_level["batch_size"],
                new_level["accum_steps"],
            )

    def get_max_length(self):
        return int(self.level_actual["max_len"])

    def get_batch_size(self):
        return int(self.level_actual["batch_size"])

    def get_accum_steps(self):
        return int(self.level_actual["accum_steps"])

    def __iter__(self):
        # Compatibilidade de API. A geração dos exemplos é feita pelo
        # StreamingTranslationDataset.
        return iter(())

    def __len__(self):
        return 0


class DynamicBatchSampler(torch.utils.data.BatchSampler):
    """
    Mantém a API do DynamicBatchSampler original.

    No modo streaming, a montagem física dos batches acontece em
    CurriculumBatchIterableDataset. Esta classe centraliza o estado do
    curriculum e expõe batch_size/max_len/accum_steps.
    """

    def __init__(self, sampler: CurriculumLengthSampler):
        self.sampler = sampler
        self.drop_last = False
        self.batch_size = self.sampler.get_batch_size()

    def step(self):
        self.sampler.step()
        self.batch_size = self.sampler.get_batch_size()

    def get_max_length(self):
        return self.sampler.get_max_length()

    def get_batch_size(self):
        return self.sampler.get_batch_size()

    def get_accum_steps(self):
        return self.sampler.get_accum_steps()

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0


class DynamicCollator:
    """Padding dinâmico usando o max_len atual do curriculum."""

    def __init__(self, batch_sampler=None, curriculum_sampler=None, pad_value: int = 0):
        self.batch_sampler = batch_sampler
        self.curriculum_sampler = curriculum_sampler
        self.pad_value = pad_value

    def get_max_length(self):
        if self.batch_sampler is not None:
            return self.batch_sampler.get_max_length()
        if self.curriculum_sampler is not None:
            return self.curriculum_sampler.get_max_length()
        raise RuntimeError("Nenhum sampler de curriculum configurado.")

    def __call__(self, batch):
        if not batch:
            raise ValueError("Batch vazio recebido pelo DynamicCollator.")

        max_len = self.get_max_length()

        srcs = []
        tgts = []

        for src, tgt in batch:
            src_list = src.tolist() if isinstance(src, torch.Tensor) else list(src)
            tgt_list = tgt.tolist() if isinstance(tgt, torch.Tensor) else list(tgt)

            src_list = src_list[:max_len]
            tgt_list = tgt_list[:max_len]

            src_list += [self.pad_value] * (max_len - len(src_list))
            tgt_list += [self.pad_value] * (max_len - len(tgt_list))

            srcs.append(torch.tensor(src_list, dtype=torch.long))
            tgts.append(torch.tensor(tgt_list, dtype=torch.long))

        return torch.stack(srcs), torch.stack(tgts)



class FixedShardCache:
    """
    Cache LRU de shards binários fixos.

    Formato dos shards gerados pelo tokenizador:
        [256 int32 SRC][256 int32 TGT][int32 src_length][int32 tgt_length]

    Cada registro ocupa 2056 bytes.
    O metadata informa src_length/tgt_length, e somente os tokens reais
    são devolvidos ao Dataset.
    """

    MAX_SRC_LENGTH = 512
    MAX_TGT_LENGTH = 512
    DTYPE = np.dtype("<i4")
    RECORD_INTS = MAX_SRC_LENGTH + MAX_TGT_LENGTH + 2
    RECORD_BYTES = RECORD_INTS * DTYPE.itemsize

    def __init__(self, cache_size=5):
        self.cache_size = max(1, int(cache_size))
        self.cache = {}
        self.cache_order = []

    def _resolve_path(self, shard_path, metadata_path=None):
        path = os.path.abspath(os.path.expanduser(str(shard_path)))

        if os.path.isfile(path):
            return path

        if metadata_path:
            candidate = os.path.abspath(
                os.path.join(
                    os.path.dirname(os.path.abspath(metadata_path)),
                    str(shard_path),
                )
            )
            if os.path.isfile(candidate):
                return candidate

        raise FileNotFoundError(f"Shard não encontrado: {shard_path}")

    def get(self, shard_path, metadata_path=None):
        path = self._resolve_path(shard_path, metadata_path)

        if path in self.cache:
            self.cache_order.remove(path)
            self.cache_order.append(path)
            return self.cache[path]

        logger.info("Mapeando shard: %s", path)

        data = np.memmap(
            path,
            dtype=self.DTYPE,
            mode="r",
        )

        if data.size % self.RECORD_INTS != 0:
            raise RuntimeError(
                f"Shard inválido: {path}. "
                f"O tamanho não é múltiplo de {self.RECORD_INTS} int32 "
                f"({self.RECORD_BYTES} bytes por registro)."
            )

        self.cache[path] = data
        self.cache_order.append(path)

        while len(self.cache_order) > self.cache_size:
            old_path = self.cache_order.pop(0)
            old_data = self.cache.pop(old_path, None)
            del old_data

        return data

    def get_example(
        self,
        shard_path,
        shard_index,
        src_length,
        tgt_length,
        metadata_path=None,
    ):
        data = self.get(shard_path, metadata_path)

        shard_index = int(shard_index)
        src_length = int(src_length)
        tgt_length = int(tgt_length)

        if not 0 <= src_length <= self.MAX_SRC_LENGTH:
            raise RuntimeError(
                f"src_length inválido: {src_length}; "
                f"esperado 0..{self.MAX_SRC_LENGTH}"
            )

        if not 0 <= tgt_length <= self.MAX_TGT_LENGTH:
            raise RuntimeError(
                f"tgt_length inválido: {tgt_length}; "
                f"esperado 0..{self.MAX_TGT_LENGTH}"
            )

        if shard_index < 0:
            raise RuntimeError(f"shard_index inválido: {shard_index}")

        base = shard_index * self.RECORD_INTS
        end = base + self.RECORD_INTS

        if end > data.size:
            raise RuntimeError(
                f"shard_index={shard_index} está fora do shard "
                f"{shard_path}."
            )

        record = data[base:end]

        # Layout físico:
        # [256 SRC][256 TGT][src_length][tgt_length]
        stored_src_length = int(record[self.RECORD_INTS - 2])
        stored_tgt_length = int(record[self.RECORD_INTS - 1])

        if (
            stored_src_length != src_length
            or stored_tgt_length != tgt_length
        ):
            raise RuntimeError(
                "Metadata e shard estão inconsistentes: "
                f"metadata=({src_length}, {tgt_length}), "
                f"shard=({stored_src_length}, {stored_tgt_length}), "
                f"shard={shard_path}, index={shard_index}."
            )

        src = np.asarray(
            record[:self.MAX_SRC_LENGTH][:src_length],
            dtype=np.int64,
        )
        tgt = np.asarray(
            record[
                self.MAX_SRC_LENGTH:
                self.MAX_SRC_LENGTH + self.MAX_TGT_LENGTH
            ][:tgt_length],
            dtype=np.int64,
        )

        return src, tgt

    def close(self):
        self.cache.clear()
        self.cache_order.clear()


class TranslationDataset(Dataset):
    """
    Dataset indexável usando o metadata inteiro carregado na RAM.

    Os tokens ficam nos shards binários FIXOS. O metadata informa:
        shard_path
        shard_index
        src_length
        tgt_length

    O shard sempre possui espaço físico para 256 tokens em cada lado, mas
    somente os tokens até src_length/tgt_length são devolvidos ao modelo.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        split_name: str,
        curriculum_sampler: CurriculumLengthSampler,
        metadata_path: str,
        invert_src: bool = False,
        max_len: int | None = None,
        cache_size: int = 3,
        shuffle: bool = False,
        seed: int = 42,
    ):
        super().__init__()

        self.metadata = metadata
        self.split_name = split_name.lower()
        self.curriculum_sampler = curriculum_sampler
        self.metadata_path = metadata_path
        self.invert_src = invert_src
        self.absolute_max_len = max_len
        self.cache = FixedShardCache(cache_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

        split_mask = (
            self.metadata["split"].astype(str).str.lower() == self.split_name
        )
        self.indices = np.flatnonzero(split_mask.to_numpy())

        if len(self.indices) == 0:
            raise RuntimeError(
                f"Nenhum exemplo encontrado para split='{self.split_name}'."
            )

    def __len__(self):
        return len(self.indices)

    def _get_current_max_len(self):
        curriculum_max = self.curriculum_sampler.get_max_length()
        if self.absolute_max_len is None:
            return curriculum_max
        return min(curriculum_max, self.absolute_max_len)

    def __getitem__(self, idx):
        row_idx = int(self.indices[idx])
        row = self.metadata.iloc[row_idx]

        shard_path = str(row["shard_path"])
        shard_index = int(row["shard_index"])

        # IMPORTANTE:
        # O tamanho REAL vem do metadata, e não dos 256 slots físicos.
        src_length = int(row["src_length"])
        tgt_length = int(row["tgt_length"])

        max_len = self._get_current_max_len()

        # Curriculum filtra exemplos que ainda estão acima do limite atual.
        # Não há necessidade de ler os tokens para descobrir isso.
        if src_length > max_len or tgt_length > max_len:
            raise IndexError(
                f"Exemplo acima do max_len atual: "
                f"src_length={src_length}, tgt_length={tgt_length}, "
                f"max_len={max_len}."
            )

        src, tgt = self.cache.get_example(
            shard_path=shard_path,
            shard_index=shard_index,
            src_length=src_length,
            tgt_length=tgt_length,
            metadata_path=self.metadata_path,
        )

        # Cópias pequenas: apenas os tokens reais, nunca os 256 slots.
        src = np.array(src, dtype=np.int64, copy=True)
        tgt = np.array(tgt, dtype=np.int64, copy=True)

        if self.invert_src:
            src = src[::-1].copy()

        return torch.from_numpy(src), torch.from_numpy(tgt)

    def close(self):
        self.cache.close()


class CurriculumBatchIterableDataset(torch.utils.data.IterableDataset):
    """
    Mantém somente a responsabilidade de montar batches dinâmicos.

    O metadata já está inteiro na RAM; portanto não existe mais streaming do
    Parquet nem necessidade de múltiplos workers para ler metadata.
    """

    def __init__(self, dataset: TranslationDataset, curriculum_sampler):
        super().__init__()
        self.dataset = dataset
        self.curriculum_sampler = curriculum_sampler

    def __len__(self):
        return math.ceil(
            len(self.dataset) / self.curriculum_sampler.get_batch_size()
        )

    def __iter__(self):
        batch = []

        # Embaralhamos os chunks, não cada exemplo individualmente.
        # Isso preserva a localidade dos NPZs de 20k exemplos e evita
        # abrir/trocar milhares de arquivos repetidamente no cache.
        # Os índices abaixo são POSICIONAIS dentro de self.dataset.
        # Isso é importante porque self.dataset.__getitem__ já converte
        # a posição local para a linha correspondente do DataFrame.
        indices = np.arange(len(self.dataset), dtype=np.int64)

        if self.dataset.shuffle:
            global_indices = self.dataset.indices[indices]
            paths = self.dataset.metadata.iloc[global_indices]["shard_path"].to_numpy()

            # Agrupamos por shard e embaralhamos os grupos. Isso preserva
            # localidade do cache/memmap sem exigir que o metadata saia da RAM.
            boundaries = np.flatnonzero(paths[1:] != paths[:-1]) + 1
            groups = np.split(indices, boundaries)

            rng = np.random.default_rng(self.dataset.seed)
            rng.shuffle(groups)

            indices = np.concatenate(groups) if groups else indices

        for idx in indices:
            batch_size = self.curriculum_sampler.get_batch_size()

            try:
                example = self.dataset[int(idx)]
            except IndexError:
                # Exemplo acima do limite do curriculum atual.
                continue

            batch.append(example)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def load_metadata_to_ram(metadata_path: str) -> pd.DataFrame:
    """Carrega o único metadata Parquet inteiro para a RAM."""
    logger.info("Carregando metadata inteiro na RAM: %s", metadata_path)
    metadata = pd.read_parquet(metadata_path)

    required = {"split", "shard_path", "shard_index", "src_length", "tgt_length"}
    missing = required.difference(metadata.columns)
    if missing:
        raise RuntimeError(
            "Metadata não possui as colunas obrigatórias: "
            f"{sorted(missing)}"
        )

    logger.info(
        "Metadata carregado: %d linhas, %.2f MB em memória.",
        len(metadata),
        metadata.memory_usage(deep=True).sum() / (1024 ** 2),
    )
    return metadata


def build_dataset(
    metadata: pd.DataFrame,
    split_name: str,
    curriculum_sampler: CurriculumLengthSampler,
    metadata_path: str,
    invert_src: bool = False,
    max_len: int | None = None,
    cache_size: int = 2,
    shuffle: bool = False,
    seed: int = 42,
):
    dataset = TranslationDataset(
        metadata=metadata,
        split_name=split_name,
        curriculum_sampler=curriculum_sampler,
        metadata_path=metadata_path,
        invert_src=invert_src,
        max_len=max_len,
        cache_size=cache_size,
        shuffle=shuffle,
        seed=seed,
    )

    logger.info(
        "Split '%s': %d exemplos disponíveis na RAM.",
        split_name,
        len(dataset),
    )

    return CurriculumBatchIterableDataset(
        dataset=dataset,
        curriculum_sampler=curriculum_sampler,
    )

def build_collate_fn(pad_id: int = 0):
    def collate_fn(batch):
        # batch = [(src, tgt), ...]
        max_src_len = max(len(src) for src, _ in batch)
        max_tgt_len = max(len(tgt) for _, tgt in batch)

        srcs = []
        tgts = []
        for src, tgt in batch:
            src = src[:max_src_len]
            tgt = tgt[:max_tgt_len]

            src_pad = torch.full((max_src_len,), pad_id, dtype=torch.long)
            tgt_pad = torch.full((max_tgt_len,), pad_id, dtype=torch.long)

            src_pad[: src.numel()] = src
            tgt_pad[: tgt.numel()] = tgt

            srcs.append(src_pad)
            tgts.append(tgt_pad)

        return torch.stack(srcs), torch.stack(tgts)

    return collate_fn


def eval(model, dataloader, criterion, step_info, writer, train_config, tokenizer=None):
    model.eval()
    printed_example = False
    num_batches = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="Evaluating", unit="batch")

        for src, tgt in progress:
            num_batches += 1

            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])


            loss, loss_no_tf = model.eval_step(src, tgt, criterion)

            step_info["loss_eval"] += loss
            step_info["loss_no_tf_eval"] += loss_no_tf

            if not printed_example:
                index_to_print = np.random.randint(0, src.size(0))
                preds_logits = model(
                    src[index_to_print:index_to_print + 1],
                    tgt[index_to_print:index_to_print + 1, :-1],
                )
                preds_ids = preds_logits.argmax(dim=-1)
                preds_ids_no_tf = model.predict(
                    src[index_to_print:index_to_print + 1],
                    5,
                    6,
                    max_len=tgt.size(1),
                    input_decoder=tgt[index_to_print:index_to_print + 1, :2]
                )
                ref_ids = tgt[index_to_print, 1:]

                print('Começo dos tokens de referência (tgt):', tgt[index_to_print, :2])
                print(f"Example tgt ids passed to model: {tgt[index_to_print, :-1]}")
                print(f"Example ref ids: {ref_ids}")
                print(f"Example pred ids: {preds_ids.squeeze()}")
                print(f"Example pred ids (no TF): {preds_ids_no_tf}")

                if tokenizer is not None:
                    try:
                        ref_np = ref_ids.squeeze().cpu().numpy()
                        pred_np = preds_ids.squeeze().cpu().numpy()
                        preds_no_tf_np = preds_ids_no_tf

                        print(
                            "Src decoded:",
                            tokenizer.decode(
                                src[index_to_print].cpu().numpy(),
                                skip_special_tokens=False,
                            ),
                        )
                        print(
                            f"Example ref decoded: {tokenizer.decode(ref_np, skip_special_tokens=False)}"
                        )
                        print(
                            f"Example pred decoded: {tokenizer.decode(pred_np, skip_special_tokens=False)}"
                        )
                        print(
                            "Example pred decoded (no TF): "
                            f"{tokenizer.decode_batch(preds_no_tf_np, skip_special_tokens=False)}"
                        )
                    except Exception:
                        logger.exception("Failed to decode tokens with tokenizer")

                printed_example = True

    return num_batches


def init_params(model: nn.Module, weight_init_method: str = None, bias_init_method: str = None):
    for name, param in model.named_parameters():
        if "weight" in name:
            if weight_init_method == "xavier_uniform":
                nn.init.xavier_uniform_(param)
            elif weight_init_method == "xavier_normal":
                nn.init.xavier_normal_(param)
        if "bias" in name and bias_init_method is not None:
            if bias_init_method == "zeros":
                nn.init.zeros_(param)
            elif bias_init_method == "ones":
                nn.init.ones_(param)


def log_gradients(model: nn.Module, step_info: dict, writer: SummaryWriter):
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), step_info["global_step"])
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, step_info["global_step"])


def save_configs(args, extra: dict | None = None):
    os.makedirs("configs", exist_ok=True)
    payload = vars(args).copy()
    if extra is not None:
        payload.update(extra)
    with open(f"configs/{args.name}_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)


def log_lr(optimizer: torch.optim.Optimizer, step_info: dict, writer: SummaryWriter):
    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step_info["global_step"])


def log_teacher_forcing_ratio(scheduler_sampling: SigmoidSchedulerSampling, step_info: dict, writer: SummaryWriter):
    if scheduler_sampling is not None:
        writer.add_scalar("teacher_forcing_ratio", scheduler_sampling.get_ratio(), step_info["global_step"])


def train(
    model,
    dataloader,
    dataloader_eval,
    criterion,
    optimizer,
    train_config,
    step_info,
    writer,
    early_stopper,
    scheduler,
    tokenizer=None,
    scheduler_sampling=None,
    model_name: str = "model",
):
    stop = False
    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)

    for _ in range(int(1e6)):
        base_dataset = getattr(dataloader.dataset, "dataset", dataloader.dataset)
        total_examples = len(base_dataset) if hasattr(base_dataset, "__len__") else None

        pbar = tqdm(
            total=total_examples,
            desc="Treinamento",
            unit="ex",
            mininterval=1.0,
        )

        for src, tgt in dataloader:
            pbar.update(src.size(0))
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])

            if tokenizer:
                logger.debug(f"Src tokens: {src[0]}")
                logger.debug(f"Src: {tokenizer.decode(src[0].cpu().numpy(), False)}")
                logger.debug(f"Tgt tokens: {tgt[0, :-1]}")
                logger.debug(f"Tgt: {tokenizer.decode(tgt[0].cpu().numpy(), False)}")
                logger.debug(f"Tgt tokens shifted: {tgt[0, 1:]}")
                logger.debug(f"Tgt shifted: {tokenizer.decode(tgt[0, 1:].cpu().numpy(), False)}")

            current_accum_steps = max(
                1,
                int(
                    train_config.get(
                        "accum_steps",
                        train_config["curriculum_batch_sampler"].get_accum_steps()
                        if train_config.get("curriculum_batch_sampler") is not None
                        else 1,
                    )
                ),
            )

            loss = model.train_step(
                src,
                tgt,
                criterion,
                train_config["teacher_forcing"],
                scheduler_sampling=scheduler_sampling,
            )
            loss = loss / current_accum_steps
            loss.backward()

            step_info["loss_train"] += loss.item()
            step_info["loss_train_count"] += 1
            step_info["step"] += 1

            if train_config["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config["clip_grad"])

            step_info["accum_counter"] = step_info.get("accum_counter", 0) + 1

            if step_info["accum_counter"] >= current_accum_steps:
                optimizer.step()
                step_info["accum_counter"] = 0
                scheduler.step()
                if scheduler_sampling is not None:
                    scheduler_sampling.step()

                # Avança o curriculum a cada optimizer step.
                if train_config.get("curriculum_batch_sampler") is not None:
                    train_config["curriculum_batch_sampler"].step()

                # O batch/accum_steps usados no próximo ciclo vêm do novo nível.
                if train_config.get("curriculum_batch_sampler") is not None:
                    train_config["accum_steps"] = (
                        train_config["curriculum_batch_sampler"].get_accum_steps()
                    )

                step_info["global_step"] += 1

                if step_info["global_step"] % train_config["log_steps"] == 0:
                    avg_loss = step_info["loss_train"] / max(step_info["loss_train_count"], 1)
                    logger.debug(f'Step {step_info["global_step"]} - Loss: {avg_loss:.4f}')
                    writer.add_scalar("Loss/Train", avg_loss, step_info["global_step"])
                    log_gradients(model, step_info, writer)
                    step_info["loss_train"] = 0
                    step_info["loss_train_count"] = 0

                    log_lr(optimizer, step_info, writer)
                    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)

                optimizer.zero_grad()

                if step_info["global_step"] % train_config["eval_steps"] == 0:
                    eval_batches = eval(
                        model,
                        dataloader_eval,
                        criterion,
                        step_info,
                        writer,
                        train_config,
                        tokenizer,
                    )
                    eval_loss = step_info["loss_eval"] / max(eval_batches, 1)
                    eval_loss_no_tf = step_info["loss_no_tf_eval"] / max(eval_batches, 1)
                    logger.debug(f'Step {step_info["global_step"]} - Eval Loss: {eval_loss:.4f}')
                    writer.add_scalar("Loss/Eval", eval_loss, step_info["global_step"])
                    writer.add_scalar("Loss/Eval_no_teacher_forcing", eval_loss_no_tf, step_info["global_step"])
                    step_info["loss_eval"] = 0
                    step_info["loss_no_tf_eval"] = 0

                    if eval_loss < step_info["best_eval_loss"]:
                        step_info["best_eval_loss"] = eval_loss
                        torch.save(model.state_dict(), f"artifacts/model_{model_name}.pt")
                        logger.debug(f"New best model saved at step {step_info['global_step']} with eval loss {eval_loss:.4f}")

                    stop = early_stopper.step(eval_loss)
                    if stop:
                        logger.info("Early stopping triggered.")
                        break

            pbar.close()

            if stop:
                break

        if step_info["global_step"] >= train_config["max_steps"]:
            logger.info("Max steps reached. Ending training.")
            break


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tokenized_dir", type=str, default="/media/alvarinho/dados/Datasets/refined/traducao/tokenized")
    args.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help=(
            "Caminho do único metadata Parquet consolidado. "
            "Por padrão: <tokenized_dir>/analise_textos_tokenized_split.parquet"
        ),
    )
    args.add_argument("--tokenizer_path", type=str, default="artifacts/tokenizer_en_pt_es_60000.json")
    args.add_argument("--invert_src", default=False, action="store_true")
    args.add_argument("--architecture", type=str, default="lstm", choices=["lstm", "transformer"])
    args.add_argument("--embedding_dim", type=int, default=256)
    args.add_argument("--encoder_hidden_dim", type=int, default=512)
    args.add_argument("--decoder_hidden_dim", type=int, default=512)
    args.add_argument("--encoder_num_layers", type=int, default=2)
    args.add_argument("--decoder_num_layers", type=int, default=2)
    args.add_argument("--encoder_num_heads", type=int, default=8)
    args.add_argument("--decoder_num_heads", type=int, default=8)
    args.add_argument("--encoder_dropout", type=float, default=0.5)
    args.add_argument("--decoder_dropout", type=float, default=0.5)
    args.add_argument("--encoder_bidirectional", default=False, action="store_true")
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument(
        "--shard_cache_size",
        "--npz_cache_size",
        dest="shard_cache_size",
        type=int,
        default=6,
        help="Quantidade máxima de shards binários mapeados no cache. "
             "O alias --npz_cache_size é mantido por compatibilidade.",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed usada para embaralhar a ordem dos chunks de metadata.",
    )
    args.add_argument(
        "--curriculum_levels",
        type=str,
        default=None,
        help=(
            "JSON com os níveis do curriculum. Ex.: "
            '[{"max_len":32,"batch_size":128,"accum_steps":1,"max_step":10000}, ...]'
        ),
    )
    args.add_argument("--accum_steps", type=int, default=1)
    args.add_argument("--log_steps", type=int, default=500)
    args.add_argument("--save_steps", type=int, default=5000)
    args.add_argument("--eval_steps", type=int, default=2500)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args.add_argument("--desc", type=str, default="")
    args.add_argument("--name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.add_argument("--early_stop_patience", type=int, default=100)
    args.add_argument("--early_stop_min_delta", type=float, default=0.0)
    args.add_argument("--warmup_steps", type=int, default=5000)
    args.add_argument("--max_steps", type=int, default=500000)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--vocab_size", type=int, default=120000)
    args.add_argument("--max_len", type=int, default=256)
    args.add_argument("--clip_grad", type=float, default=None)
    args.add_argument("--init_weight_method", type=str, default=None, choices=["xavier_uniform", "xavier_normal"])
    args.add_argument("--init_bias_method", type=str, default=None, choices=["zeros", "ones"])
    args.add_argument("--no-teacher_forcing", dest="teacher_forcing", default=True, action="store_false")
    args.add_argument("--scheduler_sampling", default=False, action="store_true")
    args.add_argument("--teacher_forcing_ratio", type=float, default=1.0)
    args.add_argument("--max_steps_scheduler_sampling", type=int, default=50000)
    args.add_argument("--attention", default=False, action="store_true")
    args.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing value for the loss function (default: 0.0)")
    args.add_argument("--separate_embedding", default=False, action="store_true")
    args = args.parse_args()

    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info(f"Starting training - {args.desc}")

    if args.max_len is not None:
        logger.info(
            "Teto absoluto de comprimento: %s tokens.",
            args.max_len,
        )

    # Curriculum. Se --curriculum_levels for informado, usamos exatamente
    # os níveis fornecidos. Caso contrário, usamos um curriculum padrão.
    if args.curriculum_levels:
        try:
            curriculum_levels = json.loads(args.curriculum_levels)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "--curriculum_levels precisa ser um JSON válido."
            ) from exc
    else:
        curriculum_levels = [
            {
                "max_len": min(20, args.max_len) if args.max_len is not None else 20,
                "batch_size": 128,
                "accum_steps": 1,
                "max_step": 25000,
            },
            {
                "max_len": min(64, args.max_len) if args.max_len is not None else 64,
                "batch_size": 128,
                "accum_steps": 1,
                "max_step": 50000,
            },
            {
                "max_len": min(128, args.max_len) if args.max_len is not None else 128,
                "batch_size": 64,
                "accum_steps": 2,
                "max_step": 75000,
            },
            {
                "max_len": args.max_len if args.max_len is not None else 256,
                "batch_size": args.batch_size,
                "accum_steps": args.accum_steps,
                "max_step": args.max_steps,
            },
        ]

    required_keys = {"max_len", "batch_size", "accum_steps", "max_step"}
    for i, level in enumerate(curriculum_levels):
        missing = required_keys.difference(level)
        if missing:
            raise ValueError(
                f"Curriculum nível {i} está sem as chaves: {sorted(missing)}"
            )

        if int(level["max_len"]) <= 0:
            raise ValueError(f"Curriculum nível {i}: max_len deve ser > 0.")
        if int(level["batch_size"]) <= 0:
            raise ValueError(f"Curriculum nível {i}: batch_size deve ser > 0.")
        if int(level["accum_steps"]) <= 0:
            raise ValueError(f"Curriculum nível {i}: accum_steps deve ser > 0.")

    curriculum_sampler = CurriculumLengthSampler(curriculum_levels)
    dynamic_batch_sampler = DynamicBatchSampler(curriculum_sampler)

    metadata_path = resolve_metadata_path(
        args.tokenized_dir,
        args.metadata_path,
    )

    # O metadata agora cabe na RAM: carregamos uma única vez e reutilizamos
    # o mesmo DataFrame para train e val.
    metadata = load_metadata_to_ram(metadata_path)


    # ============================================================
    # LIMITAR QUANTIDADE DE OBSERVAÇÕES
    # ============================================================

    if MAX_TRAIN_OBSERVATIONS is not None:
        train_mask = metadata["split"] == "train"
        train_indices = metadata.index[train_mask][:MAX_TRAIN_OBSERVATIONS]

        # Mantém todos os outros splits e limita somente o train
        metadata = metadata[
            (~train_mask) | metadata.index.isin(train_indices)
        ]

    if MAX_VAL_OBSERVATIONS is not None:
        val_mask = metadata["split"] == "val"
        val_indices = metadata.index[val_mask][:MAX_VAL_OBSERVATIONS]

        # Mantém todos os outros splits e limita somente o val
        metadata = metadata[
            (~val_mask) | metadata.index.isin(val_indices)
        ]

    logger.info(
        "Metadata após limite: %d observações.",
        len(metadata),
    )

    for split_name in metadata["split"].unique():
        logger.info(
            "Split '%s': %d observações.",
            split_name,
            (metadata["split"] == split_name).sum(),
        )

    train_dataset = build_dataset(
        metadata=metadata,
        split_name="train",
        curriculum_sampler=curriculum_sampler,
        metadata_path=metadata_path,
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.shard_cache_size,
        shuffle=True,
        seed=args.seed,
    )

    val_dataset = build_dataset(
        metadata=metadata,
        split_name="val",
        curriculum_sampler=curriculum_sampler,
        metadata_path=metadata_path,
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.shard_cache_size,
        shuffle=False,
        seed=args.seed,
    )

    dynamic_collator = DynamicCollator(
        batch_sampler=dynamic_batch_sampler,
        pad_value=0,
    )

    # Metadata está na RAM e a leitura dos NPZs é feita localmente via cache.
    # Não precisamos de workers para dividir a leitura do metadata.
    loader_kwargs = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": 0,
        "collate_fn": dynamic_collator,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(val_dataset, **loader_kwargs)

    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    if args.architecture == "transformer":
        model = Transformer(
            embedding_dim=args.embedding_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            encoder_num_heads=args.encoder_num_heads,
            decoder_num_heads=args.decoder_num_heads,
            encoder_num_layers=args.encoder_num_layers,
            decoder_num_layers=args.decoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            vocab_size=args.vocab_size,
            kv_cache=False,
            cross_attn_cache=False,
            pad_idx=0,
            separate_embedding=args.separate_embedding,
        )
    else:
        model = LSTM(
            embedding_dim=args.embedding_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            decoder_hidden_dim=args.decoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
            decoder_num_layers=args.decoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout,
            encoder_bidirectional=args.encoder_bidirectional,
            vocab_size=args.vocab_size,
            pad_idx=0,
            attention=args.attention,
        )

    model = model.to(args.device)
    init_params(model, weight_init_method=args.init_weight_method, bias_init_method=args.init_bias_method)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)
    scheduler_sampling = SigmoidSchedulerSampling(
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        max_steps=args.max_steps_scheduler_sampling,
        use=args.scheduler_sampling,
    ) if args.scheduler_sampling else None

    train_config = {
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "device": args.device,
        "max_steps": args.max_steps,
        "clip_grad": args.clip_grad,
        "teacher_forcing": args.teacher_forcing,
        "curriculum_batch_sampler": dynamic_batch_sampler,
    }

    step_info = {
        "loss_train": 0,
        "loss_train_count": 0,
        "loss_eval": 0,
        "loss_no_tf_eval": 0,
        "step": 0,
        "best_eval_loss": float("inf"),
        "global_step": 0,
        "accum_counter": 0,
    }

    writer = SummaryWriter(log_dir=f"runs/{args.name}")
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
    save_configs(
        args,
        extra={
            "tokenized_dir": args.tokenized_dir,
            "tokenizer_path": args.tokenizer_path,
            "metadata_in_ram": True,
            "metadata_path": metadata_path,
            "shard_cache_size": args.shard_cache_size,
            "num_workers": 0,
            "curriculum_levels": curriculum_levels,
        },
    )

    logger.info("Starting training loop...")
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        train_config,
        step_info,
        writer,
        early_stopper,
        scheduler,
        tokenizer,
        scheduler_sampling,
        model_name=args.name,
    )
