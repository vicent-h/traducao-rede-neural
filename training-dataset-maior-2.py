from argparse import ArgumentParser
from datetime import datetime
import json
import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, BatchSampler
from tokenizers import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer

from models.lstm_proj_diff import LSTM
from models.transformer import Transformer
from utils.earlystopper import EarlyStopper
from utils.scheduler_sampling import SigmoidSchedulerSampling
from utils.warmup import WarmupScheduler

logger = logging.getLogger(__name__)

np.random.seed(42)


def resolve_metadata_path(tokenized_dir: str, metadata_path: str | None = None) -> str:
    """Resolve o único Parquet de metadados da nova estrutura."""
    if metadata_path:
        path = os.path.abspath(metadata_path)
    else:
        path = os.path.join(
            os.path.abspath(tokenized_dir),
            "analise_textos_tokenized.parquet",
        )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Parquet de metadados não encontrado: {path}. "
            "Use --metadata_path para informar outro caminho."
        )
    return path


class TranslationDataset(Dataset):
    """Dataset que lê exemplos dos blocos .bin via numpy.memmap."""

    def __init__(
        self, metadata_df, tokenized_dir, split_name,
        invert_src=False, max_len=None, max_examples=None,
        memmap_cache_size=4,
    ):
        required = {
            "split", "block_id", "src_offset", "tgt_offset",
            "src_length", "tgt_length",
        }
        missing = required - set(metadata_df.columns)
        if missing:
            raise ValueError(
                f"Metadados sem colunas obrigatórias: {sorted(missing)}. "
                f"Encontradas: {list(metadata_df.columns)}"
            )

        self.tokenized_dir = os.path.abspath(tokenized_dir)
        self.blocks_dir = os.path.join(self.tokenized_dir, "blocks")
        self.invert_src = invert_src
        self.max_len = max_len
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.memmap_cache_size = max(1, int(memmap_cache_size))
        self._memmaps = OrderedDict()

        splits = self.metadata_df["split"].astype(str).str.lower().to_numpy()
        self.row_indices = np.flatnonzero(splits == split_name.lower())

        if max_examples is not None and len(self.row_indices) > int(max_examples):
            rng = np.random.default_rng(42)
            self.row_indices = rng.choice(
                self.row_indices, size=max(0, int(max_examples)), replace=False
            )
            self.row_indices.sort()

        logger.info("Split '%s': %s exemplos", split_name, len(self.row_indices))

    def __len__(self):
        return len(self.row_indices)

    def _get_memmap(self, block_id, side):
        key = (int(block_id), side)
        if key in self._memmaps:
            self._memmaps.move_to_end(key)
            return self._memmaps[key]

        suffix = "src" if side == "src" else "tgt"
        path = os.path.join(
            self.blocks_dir, f"block_{int(block_id):05d}_{suffix}.bin"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Bloco binário não encontrado: {path}")
        if os.path.getsize(path) == 0:
            raise ValueError(f"Bloco binário vazio: {path}")

        mmap = np.memmap(path, dtype=np.int32, mode="r")
        self._memmaps[key] = mmap
        while len(self._memmaps) > self.memmap_cache_size:
            self._memmaps.popitem(last=False)
        return mmap

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[int(self.row_indices[idx])]
        block_id = int(row["block_id"])
        src_offset, tgt_offset = int(row["src_offset"]), int(row["tgt_offset"])
        src_length, tgt_length = int(row["src_length"]), int(row["tgt_length"])

        if min(src_offset, tgt_offset, src_length, tgt_length) < 0:
            raise ValueError(f"Offset/comprimento negativo nos metadados: {row.to_dict()}")

        src_map = self._get_memmap(block_id, "src")
        tgt_map = self._get_memmap(block_id, "tgt")

        if src_offset + src_length > len(src_map):
            raise IndexError(
                f"Source excede bloco {block_id}: {src_offset}+{src_length}>{len(src_map)}"
            )
        if tgt_offset + tgt_length > len(tgt_map):
            raise IndexError(
                f"Target excede bloco {block_id}: {tgt_offset}+{tgt_length}>{len(tgt_map)}"
            )

        # Copia somente os tokens deste exemplo para a RAM.
        src = np.asarray(
            src_map[src_offset:src_offset + src_length], dtype=np.int64
        ).copy()
        tgt = np.asarray(
            tgt_map[tgt_offset:tgt_offset + tgt_length], dtype=np.int64
        ).copy()

        if self.invert_src:
            src = src[::-1].copy()
        if self.max_len is not None:
            src, tgt = src[:self.max_len], tgt[:self.max_len]

        return torch.from_numpy(src), torch.from_numpy(tgt)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_memmaps"] = OrderedDict()
        return state


class CurriculumLengthSampler(Sampler):
    """Seleciona e embaralha exemplos compatíveis com o nível atual."""

    def __init__(self, metadata_df, curriculum_levels, seed=42):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.curriculum_levels = curriculum_levels
        self.level_index = 0
        self.step_count = 0
        self.level_actual = curriculum_levels[0]
        self.rng = np.random.default_rng(seed)
        self.src_lengths = self.metadata_df["src_length"].to_numpy(dtype=np.int64)
        self.tgt_lengths = self.metadata_df["tgt_length"].to_numpy(dtype=np.int64)
        self._skip_empty_levels()

    def _skip_empty_levels(self):
        while (
            self.level_index < len(self.curriculum_levels) - 1
            and len(self._valid_indices()) == 0
        ):
            self.level_index += 1
            self.level_actual = self.curriculum_levels[self.level_index]
            logger.warning(
                "Nível do currículo sem exemplos elegíveis; avançando para max_len=%s.",
                self.get_max_length(),
            )

    def step(self):
        self.step_count += 1
        while (
            self.level_index < len(self.curriculum_levels) - 1
            and self.step_count >= self.curriculum_levels[self.level_index]["max_step"]
        ):
            self.level_index += 1
            self.level_actual = self.curriculum_levels[self.level_index]
            self._skip_empty_levels()
            logger.info(
                "Curriculum nível %d/%d: max_len=%s, batch_size=%s, accum_steps=%s",
                self.level_index + 1, len(self.curriculum_levels),
                self.get_max_length(), self.get_batch_size(), self.get_accum_steps(),
            )

    def get_max_length(self):
        return self.level_actual["max_len"]

    def get_batch_size(self):
        return self.level_actual["batch_size"]

    def get_accum_steps(self):
        return self.level_actual["accum_steps"]

    def _valid_indices(self):
        limit = self.get_max_length()
        if limit is None:
            return np.arange(len(self.metadata_df), dtype=np.int64)
        return np.flatnonzero(
            (self.src_lengths <= limit) & (self.tgt_lengths <= limit)
        )

    def __iter__(self):
        indices = self._valid_indices()
        self.rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return len(self._valid_indices())


class DynamicBatchSampler(BatchSampler):
    def __init__(self, sampler):
        self.sampler = sampler
        self.drop_last = False

    def step(self):
        self.sampler.step()

    def get_max_length(self):
        return self.sampler.get_max_length()

    def get_batch_size(self):
        return self.sampler.get_batch_size()

    def get_accum_steps(self):
        return self.sampler.get_accum_steps()

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) >= self.get_batch_size():
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        size = max(1, self.get_batch_size())
        return (len(self.sampler) + size - 1) // size


def build_collate_fn(pad_id=0, max_len=None):
    """Padding dinâmico pelo maior comprimento presente no batch."""
    def collate_fn(batch):
        srcs, tgts = zip(*batch)
        max_src = max((x.numel() for x in srcs), default=0)
        max_tgt = max((x.numel() for x in tgts), default=0)
        if max_len is not None:
            max_src, max_tgt = min(max_src, max_len), min(max_tgt, max_len)

        src_batch = torch.full((len(srcs), max_src), pad_id, dtype=torch.long)
        tgt_batch = torch.full((len(tgts), max_tgt), pad_id, dtype=torch.long)
        for i, (src, tgt) in enumerate(zip(srcs, tgts)):
            src, tgt = src[:max_src], tgt[:max_tgt]
            src_batch[i, :src.numel()] = src
            tgt_batch[i, :tgt.numel()] = tgt
        return src_batch, tgt_batch
    return collate_fn


def load_metadata(tokenized_dir, metadata_path=None):
    path = resolve_metadata_path(tokenized_dir, metadata_path)
    logger.info("Carregando metadados: %s", path)
    df = pd.read_parquet(path)
    required = {
        "global_row", "dataset_id", "index", "split", "block_id",
        "row_in_block", "src_offset", "tgt_offset", "src_length", "tgt_length",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Parquet incompatível com a tokenização em blocos. "
            f"Colunas ausentes: {sorted(missing)}; encontradas: {list(df.columns)}"
        )
    df["split"] = df["split"].astype(str).str.lower()
    logger.info(
        "Metadados: %s linhas, %s colunas, %s blocos",
        f"{len(df):,}", len(df.columns), df["block_id"].nunique(),
    )
    return df, path


def _length_bucket(length: int, bucket_size: int = 10) -> tuple[int, int]:
    """Faixas inclusivas: 0-10, 11-20, 21-30, ..."""
    length = max(0, int(length))
    bucket_size = max(1, int(bucket_size))
    if length <= bucket_size:
        return 0, bucket_size
    start = bucket_size + 1 + ((length - bucket_size - 1) // bucket_size) * bucket_size
    return start, start + bucket_size - 1


def _normalize_prediction_ids(prediction):
    """Normaliza a saída de model.predict para list[int], inclusive batch=1."""
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().tolist()
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    # Algumas implementações retornam [[id, id, ...]] para batch=1.
    while (
        isinstance(prediction, (list, tuple))
        and len(prediction) == 1
        and isinstance(prediction[0], (list, tuple, np.ndarray, torch.Tensor))
    ):
        prediction = prediction[0]

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().tolist()
    elif isinstance(prediction, np.ndarray):
        prediction = prediction.tolist()

    if not isinstance(prediction, (list, tuple)):
        raise TypeError(
            f"Formato inesperado retornado por model.predict: {type(prediction).__name__}"
        )

    return [int(token) for token in prediction]



def _compute_batch_token_losses(
    model: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    criterion: nn.Module,
):
    """Calcula logits e cross-entropy por token, respeitando o criterion."""
    decoder_input = tgt[:, :-1]
    target = tgt[:, 1:]

    logits = model(src, decoder_input)
    batch_size, target_width, vocab_size = logits.shape

    token_losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target.reshape(-1),
        weight=getattr(criterion, "weight", None),
        ignore_index=int(getattr(criterion, "ignore_index", 0)),
        reduction="none",
        label_smoothing=float(getattr(criterion, "label_smoothing", 0.0)),
    ).reshape(batch_size, target_width)

    ignore_index = int(getattr(criterion, "ignore_index", 0))
    valid_mask = target.ne(ignore_index)
    lengths = valid_mask.sum(dim=1)
    loss_sums = (token_losses * valid_mask).sum(dim=1)

    return target, valid_mask, lengths, loss_sums


def _update_loss_statistics(
    lengths: torch.Tensor,
    loss_sums: torch.Tensor,
    valid_mask: torch.Tensor,
    bucket_size: int,
    loss_by_length: dict,
):
    """Acumula loss global (por token) e loss média por exemplo/faixa."""
    total_loss_sum = float(loss_sums.sum().item())
    total_valid_tokens = int(valid_mask.sum().item())

    for index in range(lengths.size(0)):
        length = int(lengths[index].item())
        if length <= 0:
            continue

        bucket = _length_bucket(length, bucket_size)
        stats = loss_by_length.setdefault(
            bucket, {"loss_sum": 0.0, "examples": 0}
        )
        example_loss = loss_sums[index] / lengths[index]
        stats["loss_sum"] += float(example_loss.item())
        stats["examples"] += 1

    return total_loss_sum, total_valid_tokens


def _new_translation_bucket():
    return {
        "predictions": [],
        "references": [],
        "rouge": {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        },
    }


def _decode_example(
    model: nn.Module,
    tokenizer: Tokenizer,
    src_example: torch.Tensor,
    target_tokens: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
):
    """Gera uma tradução sem Teacher Forcing e decodifica referência/predição."""
    prediction = model.predict(
        src_example,
        bos_id,
        eos_id,
        max_len=max_len,
    )
    prediction_ids = _normalize_prediction_ids(prediction)
    reference_ids = target_tokens.detach().cpu().tolist()

    prediction_text = tokenizer.decode(
        prediction_ids, skip_special_tokens=True
    ).strip()
    reference_text = tokenizer.decode(
        reference_ids, skip_special_tokens=True
    ).strip()

    return prediction_text, reference_text


def _update_translation_statistics(
    model: nn.Module,
    tokenizer: Tokenizer,
    src: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    lengths: torch.Tensor,
    bucket_size: int,
    bos_id: int,
    eos_id: int,
    rouge_metric: rouge_scorer.RougeScorer,
    texts_by_length: dict,
):
    """Gera traduções e acumula BLEU/ROUGE por faixa de comprimento."""
    for index in range(src.size(0)):
        length = int(lengths[index].item())
        if length <= 0:
            continue

        bucket = _length_bucket(length, bucket_size)
        bucket_stats = texts_by_length.setdefault(
            bucket, _new_translation_bucket()
        )

        prediction_text, reference_text = _decode_example(
            model=model,
            tokenizer=tokenizer,
            src_example=src[index:index + 1],
            target_tokens=target[index][valid_mask[index]],
            bos_id=bos_id,
            eos_id=eos_id,
            max_len=target.size(1) + 1,
        )

        bucket_stats["predictions"].append(prediction_text)
        bucket_stats["references"].append(reference_text)

        rouge_scores = rouge_metric.score(reference_text, prediction_text)
        for metric, score in rouge_scores.items():
            bucket_stats["rouge"][metric] += score.fmeasure


def _write_loss_metrics(
    writer: SummaryWriter,
    step: int,
    total_loss_sum: float,
    total_valid_tokens: int,
    loss_by_length: dict,
):
    """Registra loss/perplexity globais e segmentadas no TensorBoard."""
    eval_loss = total_loss_sum / max(total_valid_tokens, 1)
    eval_perplexity = float(np.exp(min(eval_loss, 100.0)))

    writer.add_scalar("Loss/Eval", eval_loss, step)
    writer.add_scalar("Perplexity/Eval", eval_perplexity, step)

    logger.info(
        "Step %d | Eval loss %.5f | Perplexity %.5f | tokens %d",
        step, eval_loss, eval_perplexity, total_valid_tokens,
    )

    for (bucket_start, bucket_end), stats in sorted(loss_by_length.items()):
        bucket_name = f"{bucket_start:03d}-{bucket_end:03d}"
        bucket_loss = stats["loss_sum"] / max(stats["examples"], 1)
        bucket_perplexity = float(np.exp(min(bucket_loss, 100.0)))

        writer.add_scalar(
            f"Loss/Eval_by_length/{bucket_name}", bucket_loss, step
        )
        writer.add_scalar(
            f"Perplexity/Eval_by_length/{bucket_name}",
            bucket_perplexity,
            step,
        )
        writer.add_scalar(
            f"Examples/Eval_by_length/{bucket_name}",
            stats["examples"],
            step,
        )

        logger.info(
            "Eval length %s | loss %.5f | perplexity %.5f | examples %d",
            bucket_name, bucket_loss, bucket_perplexity, stats["examples"],
        )

    return eval_loss


def _write_translation_metrics(
    writer: SummaryWriter,
    step: int,
    texts_by_length: dict,
):
    """Calcula e registra BLEU/ROUGE globais e por faixa de comprimento."""
    all_predictions = []
    all_references = []
    global_rouge_sums = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
    }

    for bucket_stats in texts_by_length.values():
        all_predictions.extend(bucket_stats["predictions"])
        all_references.extend(bucket_stats["references"])

        for metric in global_rouge_sums:
            global_rouge_sums[metric] += bucket_stats["rouge"][metric]

    if not all_predictions:
        logger.warning("Nenhuma tradução válida foi gerada para calcular BLEU/ROUGE.")
        return

    global_bleu = sacrebleu.corpus_bleu(
        all_predictions, [all_references]
    ).score
    writer.add_scalar("Metrics/BLEU", global_bleu, step)

    example_count = len(all_predictions)
    global_rouge = {
        metric: score_sum / example_count
        for metric, score_sum in global_rouge_sums.items()
    }
    for metric, value in global_rouge.items():
        writer.add_scalar(f"Metrics/{metric.upper()}", value, step)

    logger.info(
        "Translation metrics | BLEU %.2f | ROUGE-1 %.4f | "
        "ROUGE-2 %.4f | ROUGE-L %.4f | examples %d",
        global_bleu,
        global_rouge["rouge1"],
        global_rouge["rouge2"],
        global_rouge["rougeL"],
        example_count,
    )

    for (bucket_start, bucket_end), bucket_stats in sorted(texts_by_length.items()):
        predictions = bucket_stats["predictions"]
        references = bucket_stats["references"]
        count = len(predictions)

        if count == 0:
            continue

        bucket_name = f"{bucket_start:03d}-{bucket_end:03d}"
        bucket_bleu = sacrebleu.corpus_bleu(
            predictions, [references]
        ).score

        writer.add_scalar(
            f"Metrics/BLEU_by_length/{bucket_name}",
            bucket_bleu,
            step,
        )

        bucket_rouge = {}
        for metric, score_sum in bucket_stats["rouge"].items():
            average = score_sum / count
            bucket_rouge[metric] = average
            writer.add_scalar(
                f"Metrics/{metric.upper()}_by_length/{bucket_name}",
                average,
                step,
            )

        writer.add_scalar(
            f"Examples/Metrics_by_length/{bucket_name}", count, step
        )

        logger.info(
            "Metrics length %s | BLEU %.2f | ROUGE-1 %.4f | "
            "ROUGE-2 %.4f | ROUGE-L %.4f | examples %d",
            bucket_name,
            bucket_bleu,
            bucket_rouge["rouge1"],
            bucket_rouge["rouge2"],
            bucket_rouge["rougeL"],
            count,
        )


def eval(
    model,
    dataloader,
    criterion,
    step_info,
    writer,
    train_config,
    tokenizer=None,
):
    """
    Executa a validação e registra as métricas no TensorBoard.

    A loss é calculada com Teacher Forcing. BLEU e ROUGE usam traduções
    autoregressivas, sem Teacher Forcing. O agrupamento considera o número
    de tokens válidos do target, ignorando padding e o primeiro token.
    """
    model.eval()

    device = train_config["device"]
    step = step_info["global_step"]
    bucket_size = max(1, int(train_config.get("eval_length_bucket_size", 25)))
    bos_id = int(train_config.get("bos_id", 5))
    eos_id = int(train_config.get("eos_id", 6))

    loss_by_length = {}
    texts_by_length = {}
    total_loss_sum = 0.0
    total_valid_tokens = 0

    rouge_metric = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )

    try:
        with torch.inference_mode():
            for src, tgt in tqdm(
                dataloader, desc="Evaluating", unit="batch"
            ):
                src = src.to(device, non_blocking=True)
                tgt = tgt.to(device, non_blocking=True)

                target, valid_mask, lengths, loss_sums = (
                    _compute_batch_token_losses(
                        model=model,
                        src=src,
                        tgt=tgt,
                        criterion=criterion,
                    )
                )

                batch_loss_sum, batch_valid_tokens = _update_loss_statistics(
                    lengths=lengths,
                    loss_sums=loss_sums,
                    valid_mask=valid_mask,
                    bucket_size=bucket_size,
                    loss_by_length=loss_by_length,
                )
                total_loss_sum += batch_loss_sum
                total_valid_tokens += batch_valid_tokens

                # if tokenizer is not None:
                #     _update_translation_statistics(
                #         model=model,
                #         tokenizer=tokenizer,
                #         src=src,
                #         target=target,
                #         valid_mask=valid_mask,
                #         lengths=lengths,
                #         bucket_size=bucket_size,
                #         bos_id=bos_id,
                #         eos_id=eos_id,
                #         rouge_metric=rouge_metric,
                #         texts_by_length=texts_by_length,
                #     )

        eval_loss = _write_loss_metrics(
            writer=writer,
            step=step,
            total_loss_sum=total_loss_sum,
            total_valid_tokens=total_valid_tokens,
            loss_by_length=loss_by_length,
        )

        if tokenizer is not None:
            _write_translation_metrics(
                writer=writer,
                step=step,
                texts_by_length=texts_by_length,
            )
        else:
            logger.warning(
                "Tokenizer não fornecido: BLEU e ROUGE não foram calculados."
            )

        return eval_loss

    finally:
        model.train()

def init_params(model: nn.Module, weight_init_method: str = None, bias_init_method: str = None):
    for name, param in model.named_parameters():
        if "weight" in name:
            if param.dim() >= 2:
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
    model, dataloader, dataloader_eval, criterion, optimizer,
    train_config, step_info, writer, early_stopper, scheduler,
    tokenizer=None, scheduler_sampling=None, model_name="model",
):
    stop = False
    batch_sampler = dataloader.batch_sampler
    optimizer.zero_grad(set_to_none=True)
    accumulation_counter = 0
    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)

    while step_info["global_step"] < train_config["max_steps"] and not stop:
        progress = tqdm(dataloader, total=len(dataloader), desc="Training", unit="batch")
        for src, tgt in progress:
            src = src.to(train_config["device"], non_blocking=True)
            tgt = tgt.to(train_config["device"], non_blocking=True)

            accum_steps = (
                batch_sampler.get_accum_steps()
                if hasattr(batch_sampler, "get_accum_steps")
                else train_config["accum_steps"]
            )
            accum_steps = max(1, int(accum_steps))

            loss = model.train_step(
                src, tgt, criterion, train_config["teacher_forcing"],
                scheduler_sampling=scheduler_sampling,
            )
            (loss / accum_steps).backward()
            step_info["loss_train"] += float(loss.detach())
            step_info["loss_train_count"] += 1
            step_info["step"] += 1
            accumulation_counter += 1

            if accumulation_counter >= accum_steps:

                if train_config["clip_grad"] is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=train_config["clip_grad"]
                    )
                if step_info["global_step"] % train_config["log_steps"] == 0:
                    log_gradients(model, step_info, writer)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulation_counter = 0
                scheduler.step()
                if scheduler_sampling is not None:
                    scheduler_sampling.step()
                step_info["global_step"] += 1

                # O currículo avança por atualização do otimizador, não por batch.
                if hasattr(batch_sampler, "step"):
                    batch_sampler.step()

                if step_info["global_step"] % train_config["log_steps"] == 0:
                    avg_loss = step_info["loss_train"] / max(step_info["loss_train_count"], 1)
                    logger.info(
                        "Step %d | loss %.4f | max_len=%s | batch=%s | accum=%s",
                        step_info["global_step"], avg_loss,
                        batch_sampler.get_max_length() if hasattr(batch_sampler, "get_max_length") else "-",
                        batch_sampler.get_batch_size() if hasattr(batch_sampler, "get_batch_size") else "-",
                        batch_sampler.get_accum_steps() if hasattr(batch_sampler, "get_accum_steps") else accum_steps,
                    )
                    writer.add_scalar("Loss/Train", avg_loss, step_info["global_step"])
                    
                    step_info["loss_train"] = 0
                    step_info["loss_train_count"] = 0
                    log_lr(optimizer, step_info, writer)
                    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)

                if step_info["global_step"] % train_config["eval_steps"] == 0:
                    eval_loss = eval(
                        model, dataloader_eval, criterion, step_info,
                        writer, train_config, tokenizer,
                    )

                    if eval_loss < step_info["best_eval_loss"]:
                        step_info["best_eval_loss"] = eval_loss
                        os.makedirs("artifacts", exist_ok=True)
                        torch.save(model.state_dict(), f"artifacts/model_{model_name}.pt")
                        logger.info("Melhor modelo salvo no step %d.", step_info["global_step"])

                    stop = early_stopper.step(eval_loss)
                    if stop:
                        logger.info("Early stopping acionado.")
                        break

                if step_info["global_step"] >= train_config["max_steps"]:
                    logger.info("Número máximo de steps atingido.")
                    break

        progress.close()

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), f"artifacts/model_{model_name}_last.pt")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--tokenized_dir", type=str, default="/media/alvarinho/dados/Datasets/refined/traducao/tokenized")
    args.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Caminho do único Parquet de metadados; por padrão fica dentro de tokenized_dir.",
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
    args.add_argument("--eval_batch_size", type=int, default=64)
    args.add_argument("--max_examples_val", type=int, default=10000, help="Máximo de exemplos de validação; 0 usa todos.")
    args.add_argument("--memmap_cache_size", type=int, default=12, help="Arquivos .bin mantidos em cache por worker.")
    args.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Número de workers do DataLoader.",
    )
    args.add_argument("--seed", type=int, default=42, help="Seed para embaralhar os exemplos de treino.")
    args.add_argument("--accum_steps", type=int, default=1)
    args.add_argument("--log_steps", type=int, default=500)
    args.add_argument("--save_steps", type=int, default=5000)
    args.add_argument("--eval_steps", type=int, default=2500)
    args.add_argument(
        "--eval_length_bucket_size", type=int, default=25,
        help="Largura das faixas de comprimento na validação (default: 10; faixas 0-10, 11-20, ...).",
    )
    args.add_argument("--bos_id", type=int, default=5, help="ID do token BOS usado por model.predict.")
    args.add_argument("--eos_id", type=int, default=6, help="ID do token EOS usado por model.predict.")
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args.add_argument("--desc", type=str, default="")
    args.add_argument("--name", type=str, default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.add_argument("--early_stop_patience", type=int, default=100)
    args.add_argument("--early_stop_min_delta", type=float, default=0.0)
    args.add_argument("--warmup_steps", type=int, default=5000)
    args.add_argument("--max_steps", type=int, default=500000)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--vocab_size", type=int, default=120000)
    args.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Trunca source e target neste limite; por padrão não trunca.",
    )
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
    args.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (L2 penalty) for the optimizer (default: 0.0)")
    args = args.parse_args()
    

    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    logger.info(f"Starting training - {args.desc}")

    if args.max_len is not None:
        logger.info(
            "Aplicando truncamento máximo de %s tokens por exemplo no loader.",
            args.max_len,
        )

    metadata_df, resolved_metadata_path = load_metadata(
        args.tokenized_dir, args.metadata_path
    )

    save_configs(
            args,
            extra={
                "tokenized_dir": args.tokenized_dir,
                "tokenizer_path": args.tokenizer_path,
                "streaming_metadata": False,
                "token_storage_format": "int32_binary_blocks_memmap",
                "metadata_path": resolved_metadata_path,
                "metadata_loaded_in_memory": True,
                "num_workers": args.num_workers,
            },
        )

    metadata_df_train = (
        metadata_df[metadata_df["split"] == "train"].reset_index(drop=True)
    )
    metadata_df_val = (
        metadata_df[metadata_df["split"] == "val"].reset_index(drop=True)
    )
    if metadata_df_train.empty:
        raise ValueError("Não há exemplos com split='train' no Parquet.")
    if metadata_df_val.empty:
        raise ValueError("Não há exemplos com split='val' no Parquet.")

    if args.max_examples_val > 0 and len(metadata_df_val) > args.max_examples_val:
        metadata_df_val = metadata_df_val.sample(
            n=args.max_examples_val, random_state=args.seed
        ).reset_index(drop=True)

    final_max_len = args.max_len
    if final_max_len is None:
        final_max_len = int(max(
            metadata_df_train["src_length"].max(),
            metadata_df_train["tgt_length"].max(),
        ))

    curriculum_levels = [
        {"max_step": 10000, "max_len": 25, "batch_size": 128, "accum_steps": 1},
        {"max_step": 20000, "max_len": 50, "batch_size": 64, "accum_steps": 2},
        {"max_step": 35000, "max_len": 100, "batch_size": 32, "accum_steps": 4},
        {
            "max_step": args.max_steps,
            "max_len": final_max_len,
            "batch_size": args.batch_size,
            "accum_steps": args.accum_steps,
        },
    ]
    # Evita níveis que estejam além do total de steps configurado.
    curriculum_levels = [
        level for level in curriculum_levels
        if level["max_step"] < args.max_steps
    ] + [{
        "max_step": args.max_steps,
        "max_len": final_max_len,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
    }]

    sampler = CurriculumLengthSampler(
        metadata_df_train, curriculum_levels, seed=args.seed
    )
    batch_sampler = DynamicBatchSampler(sampler)

    train_dataset = TranslationDataset(
        metadata_df_train, args.tokenized_dir, "train",
        invert_src=args.invert_src, max_len=args.max_len,
        memmap_cache_size=args.memmap_cache_size,
    )
    val_dataset = TranslationDataset(
        metadata_df_val, args.tokenized_dir, "val",
        invert_src=args.invert_src, max_len=args.max_len,
        memmap_cache_size=args.memmap_cache_size,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        generator=generator,
        batch_sampler=batch_sampler,
        collate_fn=build_collate_fn(pad_id=0, max_len=args.max_len),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=build_collate_fn(pad_id=0, max_len=args.max_len),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        "eval_length_bucket_size": args.eval_length_bucket_size,
        "bos_id": args.bos_id,
        "eos_id": args.eos_id,
        "device": args.device,
        "max_steps": args.max_steps,
        "clip_grad": args.clip_grad,
        "teacher_forcing": args.teacher_forcing,
    }

    step_info = {
        "loss_train": 0,
        "loss_train_count": 0,
        "step": 0,
        "best_eval_loss": float("inf"),
        "global_step": 0,
    }

    os.makedirs("artifacts", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{args.name}")
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
    

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