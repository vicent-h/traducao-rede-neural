from argparse import ArgumentParser
from datetime import datetime
import glob
import json
import logging
import multiprocessing as mp
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


def discover_metadata_files(tokenized_dir: str, split_name: str) -> list[str]:
    """Descobre os arquivos de metadata do split sem carregar os registros em RAM."""
    base_dir = os.path.abspath(tokenized_dir)
    meta_dir = os.path.join(base_dir, "metadata_parts")

    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"Diretório de metadata não encontrado: {meta_dir}")

    parquet_files = sorted(glob.glob(os.path.join(meta_dir, "*_meta.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"Nenhum _meta.parquet encontrado em: {meta_dir}")

    split_name = split_name.lower()
    valid_files = []

    # O filtro do split é feito por arquivo. Não mantemos nenhum DataFrame
    # consolidado em memória.
    for parquet_file in parquet_files:
        try:
            columns = pd.read_parquet(parquet_file, columns=["split"])
            if not columns.empty and columns["split"].astype(str).str.lower().eq(split_name).any():
                valid_files.append(parquet_file)
        except Exception:
            logger.warning("Falha ao inspecionar %s; ignorando arquivo.", parquet_file)

    if not valid_files:
        raise ValueError(
            f"Nenhum registro encontrado para split '{split_name}' em {meta_dir}"
        )

    return valid_files


class NPZCache:
    """Cache LRU simples para arquivos .npz."""

    def __init__(self, cache_size: int = 3):
        self.cache_size = max(1, int(cache_size))
        self.cache = {}
        self.cache_order = []

    def get(self, npz_path: str):
        if npz_path in self.cache:
            # Move para o fim: mais recentemente usado.
            self.cache_order.remove(npz_path)
            self.cache_order.append(npz_path)
            return self.cache[npz_path]

        chunk = np.load(npz_path, allow_pickle=False)
        self.cache[npz_path] = chunk
        self.cache_order.append(npz_path)

        while len(self.cache_order) > self.cache_size:
            old_path = self.cache_order.pop(0)
            old_chunk = self.cache.pop(old_path, None)
            if old_chunk is not None:
                try:
                    old_chunk.close()
                except Exception:
                    pass

        return chunk

    def close(self):
        for chunk in self.cache.values():
            try:
                chunk.close()
            except Exception:
                pass
        self.cache.clear()
        self.cache_order.clear()


class CurriculumLengthSampler(torch.utils.data.Sampler):
    """
    Curriculum por comprimento sem materializar índices em RAM.

    O nível atual fica em multiprocessing.Value para que os workers do
    DataLoader enxerguem as mudanças feitas pelo processo principal.
    """

    def __init__(self, curriculum_levels):
        if not curriculum_levels:
            raise ValueError("curriculum_levels não pode estar vazio.")

        self.curriculum_levels = curriculum_levels
        self.step_count = 0
        self._level_index = mp.Value("i", 0)

    @property
    def level_index(self):
        return self._level_index.value

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
            self._level_index.value = new_index

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


class NPZCache:
    """Cache LRU de arquivos .npz por worker."""

    def __init__(self, cache_size: int = 3):
        self.cache_size = max(1, int(cache_size))
        self.cache = {}
        self.cache_order = []

    def get(self, npz_path: str):
        if npz_path in self.cache:
            self.cache_order.remove(npz_path)
            self.cache_order.append(npz_path)
            return self.cache[npz_path]

        chunk = np.load(npz_path, allow_pickle=False)
        self.cache[npz_path] = chunk
        self.cache_order.append(npz_path)

        while len(self.cache_order) > self.cache_size:
            old_path = self.cache_order.pop(0)
            old_chunk = self.cache.pop(old_path, None)

            if old_chunk is not None:
                try:
                    old_chunk.close()
                except Exception:
                    pass

        return chunk

    def close(self):
        for chunk in self.cache.values():
            try:
                chunk.close()
            except Exception:
                pass

        self.cache.clear()
        self.cache_order.clear()


class StreamingTranslationDataset(torch.utils.data.IterableDataset):
    """
    Streaming dataset com suporte a curriculum por comprimento.

    Importante:
    - Não guarda o metadata inteiro na RAM.
    - Cada worker recebe arquivos diferentes.
    - O comprimento é obtido do metadata antes de carregar o .npz.
    - Exemplos acima do max_len atual são descartados.
    - Os exemplos válidos passam pelo shuffle buffer.
    - O batch_size é controlado pelo curriculum.
    """

    def __init__(
        self,
        metadata_files: list[str],
        split_name: str,
        curriculum_sampler: CurriculumLengthSampler,
        invert_src: bool = False,
        max_len: int | None = None,
        cache_size: int = 3,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
        max_examples: int | None = None,
    ):
        super().__init__()

        if not metadata_files:
            raise ValueError("metadata_files não pode estar vazio.")

        self.metadata_files = list(metadata_files)
        self.split_name = split_name.lower()
        self.curriculum_sampler = curriculum_sampler
        self.invert_src = invert_src

        # max_len aqui continua como teto absoluto opcional.
        self.absolute_max_len = max_len

        self.cache_size = max(1, int(cache_size))
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)
        self.max_examples = (
            None if max_examples is None else max(0, int(max_examples))
        )

    def _get_current_max_len(self):
        # Mantido para compatibilidade interna. Durante __iter__, o valor
        # efetivo é congelado no início da passagem.
        curriculum_max = self.curriculum_sampler.get_max_length()

        if self.absolute_max_len is None:
            return curriculum_max

        return min(curriculum_max, self.absolute_max_len)

    def _iter_metadata_rows(self, files):
        for parquet_file in files:
            try:
                metadata = pd.read_parquet(parquet_file)

                if metadata.empty:
                    continue

                required = {"split", "npz_path", "npz_index"}
                missing = required.difference(metadata.columns)

                if missing:
                    logger.warning(
                        "Ignorando %s: colunas ausentes %s",
                        parquet_file,
                        sorted(missing),
                    )
                    continue

                filtered = metadata[
                    metadata["split"].astype(str).str.lower() == self.split_name
                ]

                # O campo length pode ter nomes diferentes dependendo do
                # pipeline que gerou o metadata. Se existir, aproveitamos.
                length_column = None
                for candidate in (
                    "src_length",
                    "src_len",
                    "source_length",
                    "length",
                    "len_src",
                ):
                    if candidate in filtered.columns:
                        length_column = candidate
                        break

                for row in filtered.itertuples(index=False):
                    yield row, length_column

                del filtered
                del metadata

            except Exception:
                logger.exception(
                    "Falha ao ler metadata %s; ignorando.",
                    parquet_file,
                )

    @staticmethod
    def _get_row_length(row, length_column):
        if length_column is not None:
            try:
                return max(0, int(getattr(row, length_column)))
            except (TypeError, ValueError, AttributeError):
                pass

        # Compatibilidade com metadata que não possui coluna de tamanho.
        # Nesse caso, o comprimento precisa ser obtido do .npz.
        return None

    def _row_to_example(self, row, cache, max_len=None):
        npz_path = str(row.npz_path)
        npz_index = int(row.npz_index)

        chunk = cache.get(npz_path)

        src = chunk["src_tokens"][npz_index].astype(np.int64)
        tgt = chunk["tgt_tokens"][npz_index].astype(np.int64)

        if self.invert_src:
            src = src[::-1].copy()

        if max_len is None:
            max_len = self._get_current_max_len()

        src = src[:max_len]
        tgt = tgt[:max_len]

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    def _row_is_valid(self, row, metadata_length, cache, max_len=None):
        # if max_len is None:
        #     max_len = self._get_current_max_len()

        # if metadata_length is not None:
        #     return metadata_length <= max_len

        # # Fallback: precisamos abrir o .npz para descobrir o tamanho.
        # npz_path = str(row.npz_path)
        # npz_index = int(row.npz_index)

        # chunk = cache.get(npz_path)
        # src_length = len(chunk["src_tokens"][npz_index])

        return True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
            worker_files = self.metadata_files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_files = self.metadata_files[worker_id::num_workers]

        curriculum_level = self.curriculum_sampler.level_actual
        current_max_len = int(curriculum_level["max_len"])

        if self.absolute_max_len is not None:
            current_max_len = min(current_max_len, self.absolute_max_len)

        rng = np.random.default_rng(
            self.seed
            + worker_id
            + self.curriculum_sampler.level_index * 100003
        )

        cache = NPZCache(self.cache_size)
        buffer = []

        yielded_examples = 0

        try:
            for row, length_column in self._iter_metadata_rows(worker_files):

                # Limite de exemplos
                if (
                    self.max_examples is not None
                    and yielded_examples >= self.max_examples
                ):
                    break

                metadata_length = self._get_row_length(row, length_column)

                if not self._row_is_valid(
                    row,
                    metadata_length,
                    cache,
                    current_max_len,
                ):
                    continue

                if len(buffer) < self.shuffle_buffer_size:
                    buffer.append(row)
                    continue

                idx = int(rng.integers(0, len(buffer)))
                selected = buffer[idx]
                buffer[idx] = row

                yield self._row_to_example(
                    selected,
                    cache,
                    current_max_len,
                )

                yielded_examples += 1

            while buffer:

                if (
                    self.max_examples is not None
                    and yielded_examples >= self.max_examples
                ):
                    break

                idx = int(rng.integers(0, len(buffer)))
                selected = buffer.pop(idx)

                yield self._row_to_example(
                    selected,
                    cache,
                    current_max_len,
                )

                yielded_examples += 1

        finally:
            cache.close()


class CurriculumBatchIterableDataset(torch.utils.data.IterableDataset):
    """
    Camada final que transforma o fluxo de exemplos em batches dinâmicos.

    Isso evita depender de BatchSampler + Dataset indexável, que exigiria
    índices globais em memória.
    """

    def __init__(
        self,
        dataset: StreamingTranslationDataset,
        curriculum_sampler: CurriculumLengthSampler,
    ):
        super().__init__()
        self.dataset = dataset
        self.curriculum_sampler = curriculum_sampler

    def __iter__(self):
        batch = []
        batch_size = self.curriculum_sampler.get_batch_size()

        for example in self.dataset:
            batch.append(example)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def build_streaming_dataset(
    tokenized_dir: str,
    split_name: str,
    curriculum_sampler: CurriculumLengthSampler,
    invert_src: bool = False,
    max_len: int | None = None,
    cache_size: int = 3,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
    max_examples: int | None = None
):
    metadata_files = discover_metadata_files(tokenized_dir, split_name)

    logger.info(
        "Split '%s': %d arquivos de metadata serão processados por streaming.",
        split_name,
        len(metadata_files),
    )

    base_dataset = StreamingTranslationDataset(
        metadata_files=metadata_files,
        split_name=split_name,
        curriculum_sampler=curriculum_sampler,
        invert_src=invert_src,
        max_len=max_len,
        cache_size=cache_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
        max_examples=max_examples
    )

    return CurriculumBatchIterableDataset(
        dataset=base_dataset,
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
                )
                ref_ids = tgt[index_to_print, 1:]

                print(f"Example tgt ids passed to model: {tgt[index_to_print, :-1]}")
                print(f"Example ref ids: {ref_ids}")
                print(f"Example pred ids: {preds_ids.squeeze()}")
                print(f"Example pred ids (no TF): {preds_ids_no_tf}")

                if tokenizer is not None:
                    try:
                        ref_np = ref_ids.squeeze().cpu().numpy().tolist()
                        pred_np = preds_ids.squeeze().cpu().numpy().tolist()

                        # Normalize preds from no-teacher-forcing to a list of sequences
                        preds_no_tf_raw = preds_ids_no_tf

                        if isinstance(preds_no_tf_raw, torch.Tensor):
                            preds_no_tf_list = preds_no_tf_raw.cpu().numpy().tolist()
                        elif isinstance(preds_no_tf_raw, np.ndarray):
                            preds_no_tf_list = preds_no_tf_raw.tolist()
                        else:
                            preds_no_tf_list = preds_no_tf_raw

                        # If a single sequence (1D), wrap it for uniform handling
                        if preds_no_tf_list and not isinstance(preds_no_tf_list[0], (list, tuple)):
                            preds_no_tf_list = [preds_no_tf_list]

                        decoded_preds_no_tf = []
                        for seq in preds_no_tf_list:
                            try:
                                # Ensure sequence is a plain Python list of ints
                                seq_list = list(seq)
                                decoded_preds_no_tf.append(
                                    tokenizer.decode(seq_list, skip_special_tokens=False)
                                )
                            except Exception:
                                decoded_preds_no_tf.append("<decode error>")

                        print("Src decoded:", tokenizer.decode(list(src[index_to_print].cpu().numpy()), skip_special_tokens=False))
                        print(f"Example ref decoded: {tokenizer.decode(ref_np, skip_special_tokens=False)}")
                        print(f"Example pred decoded: {tokenizer.decode(pred_np, skip_special_tokens=False)}")
                        print("Example pred decoded (no TF):", "; ".join(decoded_preds_no_tf))
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
        for src, tgt in tqdm(dataloader):
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

            if stop:
                break

        if step_info["global_step"] >= train_config["max_steps"]:
            logger.info("Max steps reached. Ending training.")
            break


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tokenized_dir", type=str, default="/media/alvarinho/dados/Datasets/refined/traducao/tokenized")
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
        "--num_workers",
        type=int,
        default=2,
        help="Número de workers do DataLoader.",
    )
    args.add_argument(
        "--metadata_shuffle_buffer",
        type=int,
        default=1000,
        help="Quantidade de registros mantidos no buffer de shuffle.",
    )
    args.add_argument(
        "--npz_cache_size",
        type=int,
        default=3,
        help="Quantidade máxima de arquivos .npz mantidos no cache por worker.",
    )
    args.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed do shuffle do metadata.",
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
    args.add_argument("--max_steps_scheduler_sampling", type=int, default=500000)
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
                "max_len": min(30, args.max_len) if args.max_len is not None else 32,
                "batch_size": 128,
                "accum_steps": 1,
                "max_step": 10000,
            },
            {
                "max_len": min(64, args.max_len) if args.max_len is not None else 64,
                "batch_size": 128,
                "accum_steps": 1,
                "max_step": 25000,
            },
            {
                "max_len": min(128, args.max_len) if args.max_len is not None else 128,
                "batch_size": 64,
                "accum_steps": 2,
                "max_step": 50000,
            },
            {
                "max_len": min(256, args.max_len) if args.max_len is not None else 128,
                "batch_size": 32,
                "accum_steps": 4,
                "max_step": 60000,
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

    train_dataset = build_streaming_dataset(
        tokenized_dir=args.tokenized_dir,
        split_name="train",
        curriculum_sampler=curriculum_sampler,
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.npz_cache_size,
        shuffle_buffer_size=args.metadata_shuffle_buffer,
        seed=args.seed,
    )

    # Validação usa o mesmo nível atual do curriculum, mas não embaralha.
    val_dataset = build_streaming_dataset(
        tokenized_dir=args.tokenized_dir,
        split_name="val",
        curriculum_sampler=curriculum_sampler,
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.npz_cache_size,
        shuffle_buffer_size=args.metadata_shuffle_buffer,
        seed=args.seed,
        max_examples=1_000
    )

    dynamic_collator = DynamicCollator(
        batch_sampler=dynamic_batch_sampler,
        pad_value=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynamic_collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dynamic_collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

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
            "streaming_metadata": True,
            "metadata_shuffle_buffer": args.metadata_shuffle_buffer,
            "npz_cache_size": args.npz_cache_size,
            "num_workers": args.num_workers,
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
