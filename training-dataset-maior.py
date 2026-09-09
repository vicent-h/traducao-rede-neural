from argparse import ArgumentParser
from datetime import datetime
import glob
import json
import logging
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


class StreamingTranslationDataset(torch.utils.data.IterableDataset):
    """
    Dataset que faz streaming do metadata.

    O metadata nunca é consolidado em um DataFrame gigante. Os arquivos
    *_meta.parquet são lidos um por vez e os exemplos são entregues ao
    DataLoader.

    O shuffle é feito por um buffer em memória:
        parquet -> buffer -> amostragem aleatória -> batch

    O cache LRU mantém alguns .npz abertos para reduzir chamadas repetidas
    de np.load().
    """

    def __init__(
        self,
        metadata_files: list[str],
        split_name: str,
        invert_src: bool = False,
        max_len: int | None = None,
        cache_size: int = 3,
        shuffle_buffer_size: int = 10000,
        seed: int = 42,
    ):
        super().__init__()

        if not metadata_files:
            raise ValueError("metadata_files não pode estar vazio.")

        self.metadata_files = list(metadata_files)
        self.split_name = split_name.lower()
        self.invert_src = invert_src
        self.max_len = max_len
        self.cache_size = max(1, int(cache_size))
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.seed = int(seed)

    def _iter_metadata_rows(self, files):
        """Lê somente os arquivos atribuídos ao worker."""
        for parquet_file in files:
            try:
                # read_parquet continua materializando uma parte por vez,
                # mas nunca o dataset inteiro.
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

                for row in filtered.itertuples(index=False):
                    yield row

                # Libera a parte assim que terminamos de percorrê-la.
                del filtered
                del metadata

            except Exception:
                logger.exception("Falha ao ler metadata %s; ignorando.", parquet_file)

    def _row_to_example(self, row, cache):
        npz_path = str(row.npz_path)
        npz_index = int(row.npz_index)

        chunk = cache.get(npz_path)

        src = chunk["src_tokens"][npz_index].astype(np.int64)
        tgt = chunk["tgt_tokens"][npz_index].astype(np.int64)

        if self.invert_src:
            src = src[::-1].copy()

        if self.max_len is not None:
            src = src[: self.max_len]
            tgt = tgt[: self.max_len]

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
            worker_files = self.metadata_files
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Cada worker recebe um subconjunto diferente dos arquivos.
            worker_files = self.metadata_files[worker_id::num_workers]

        # Seed diferente por worker e por nova passagem pelo dataset.
        rng = np.random.default_rng(self.seed + worker_id)

        cache = NPZCache(self.cache_size)
        buffer = []

        try:
            row_iterator = self._iter_metadata_rows(worker_files)

            for row in row_iterator:
                # O buffer funciona como reservoir/shuffle buffer:
                # enquanto não está cheio, acumula exemplos.
                if len(buffer) < self.shuffle_buffer_size:
                    buffer.append(row)
                    continue

                # Escolhe aleatoriamente um item já acumulado.
                idx = int(rng.integers(0, len(buffer)))
                selected = buffer[idx]

                # Substitui o item retirado pelo novo.
                buffer[idx] = row

                yield self._row_to_example(selected, cache)

            # Esvazia o buffer no final da passagem.
            while buffer:
                idx = int(rng.integers(0, len(buffer)))
                selected = buffer.pop(idx)
                yield self._row_to_example(selected, cache)

        finally:
            cache.close()


def build_streaming_dataset(
    tokenized_dir: str,
    split_name: str,
    invert_src: bool = False,
    max_len: int | None = None,
    cache_size: int = 3,
    shuffle_buffer_size: int = 10000,
    seed: int = 42,
):
    metadata_files = discover_metadata_files(tokenized_dir, split_name)

    logger.info(
        "Split '%s': %d arquivos de metadata serão processados por streaming.",
        split_name,
        len(metadata_files),
    )

    return StreamingTranslationDataset(
        metadata_files=metadata_files,
        split_name=split_name,
        invert_src=invert_src,
        max_len=max_len,
        cache_size=cache_size,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=seed,
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

    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])

            loss, loss_no_tf = model.eval_step(src, tgt, criterion)

            step_info["loss_eval"] += loss
            step_info["loss_no_tf_eval"] += loss_no_tf

            if not printed_example:
                index_to_print = np.random.randint(0, src.size(0))
                preds_logits = model(src[index_to_print:index_to_print + 1], tgt[index_to_print:index_to_print + 1, :-1])
                preds_ids = preds_logits.argmax(dim=-1)
                preds_ids_no_tf = model.predict(src[index_to_print:index_to_print + 1], 5, 6, max_len=tgt.size(1))
                ref_ids = tgt[index_to_print, 1:]

                print(f"Example tgt ids passed to model: {tgt[index_to_print, :-1]}")
                print(f"Example ref ids: {ref_ids}")
                print(f"Example pred ids: {preds_ids.squeeze()}")
                print(f"Example pred ids (no TF): {preds_ids_no_tf}")

                if tokenizer is not None:
                    try:
                        ref_np = ref_ids.squeeze().cpu().numpy()
                        pred_np = preds_ids.squeeze().cpu().numpy()
                        preds_no_tf_np = preds_ids_no_tf
                        print("Src decoded:", tokenizer.decode(src[index_to_print].cpu().numpy(), skip_special_tokens=False))
                        print(f"Example ref decoded: {tokenizer.decode(ref_np, skip_special_tokens=False)}")
                        print(f"Example pred decoded: {tokenizer.decode(pred_np, skip_special_tokens=False)}")
                        print(f"Example pred decoded (no TF): {tokenizer.decode(preds_no_tf_np, skip_special_tokens=False)}")
                    except Exception:
                        logger.exception("Failed to decode tokens with tokenizer")

                printed_example = True


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

            loss = model.train_step(src, tgt, criterion, train_config["teacher_forcing"], scheduler_sampling=scheduler_sampling)
            loss = loss / train_config["accum_steps"]
            loss.backward()

            step_info["loss_train"] += loss.item()
            step_info["loss_train_count"] += 1
            step_info["step"] += 1

            if train_config["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config["clip_grad"])

            if step_info["step"] % train_config["accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                if scheduler_sampling is not None:
                    scheduler_sampling.step()

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
                    eval(model, dataloader_eval, criterion, step_info, writer, train_config, tokenizer)
                    eval_loss = step_info["loss_eval"] / max(len(dataloader_eval), 1)
                    eval_loss_no_tf = step_info["loss_no_tf_eval"] / max(len(dataloader_eval), 1)
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
        default=10000,
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
            "Aplicando truncamento máximo de %s tokens por exemplo no loader.",
            args.max_len,
        )

    train_dataset = build_streaming_dataset(
        tokenized_dir=args.tokenized_dir,
        split_name="train",
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.npz_cache_size,
        shuffle_buffer_size=args.metadata_shuffle_buffer,
        seed=args.seed,
    )

    val_dataset = build_streaming_dataset(
        tokenized_dir=args.tokenized_dir,
        split_name="val",
        invert_src=args.invert_src,
        max_len=args.max_len,
        cache_size=args.npz_cache_size,
        # Validação não precisa de shuffle.
        shuffle_buffer_size=1,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset faz o próprio shuffle.
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(pad_id=0),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=build_collate_fn(pad_id=0),
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
    }

    step_info = {
        "loss_train": 0,
        "loss_train_count": 0,
        "loss_eval": 0,
        "loss_no_tf_eval": 0,
        "step": 0,
        "best_eval_loss": float("inf"),
        "global_step": 0,
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