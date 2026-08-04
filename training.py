from argparse import ArgumentParser
from datetime import datetime
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import logging
import os
from models.lstm_proj_diff import LSTM
from models.transformer import Transformer
from utils.dataset import (
    TranslateDataset, CurriculumLengthSampler,
    DynamicBatchSampler, DynamicCollator
)
from utils.earlystopper import EarlyStopper
import pandas as pd
from logging import getLogger
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from utils.scheduler_sampling import (
    LinearSchedulerSampling, SigmoidSchedulerSampling
)
from utils.warmup import WarmupScheduler
import numpy as np

logger = getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

def eval(model: LSTM, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, step_info: dict, writer: SummaryWriter, train_config: dict, tokenizer: Tokenizer = None):
    model.eval()
    printed_example = False
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])

            loss, loss_no_tf = model.eval_step(src, tgt, criterion)

            step_info["loss_eval"] += loss
            step_info["loss_no_tf_eval"] += loss_no_tf

            if not printed_example:
                index_to_print = np.random.randint(0, src.size(0))
                # obtain model predictions (teacher-forced) and show first sample
                preds_logits: torch.Tensor = model(src[index_to_print:index_to_print+1], tgt[index_to_print:index_to_print+1, :-1])
                preds_ids = preds_logits.argmax(dim=-1)  # [batch, tgt_len]

                preds_ids_no_tf = model.predict(src[index_to_print:index_to_print+1], 5, 6, max_len=tgt.size(1))

                ref_ids: torch.Tensor = tgt[index_to_print, 1:]

                # log token ids
                print(f'Example tgt ids passed to model: {tgt[index_to_print, :-1]}')
                print(f"Example ref ids: {ref_ids}")
                print(f"Example pred ids: {preds_ids.squeeze()}")
                print(f"Example pred ids (no TF): {preds_ids_no_tf.squeeze()}")

                if tokenizer is not None:
                    try:
                        ref_np = ref_ids.squeeze().cpu().numpy()
                        pred_np = preds_ids.squeeze().cpu().numpy()
                        preds_no_tf_np = preds_ids_no_tf.squeeze().cpu().numpy()
                        print('Src decoded:', tokenizer.decode(src[index_to_print].cpu().numpy(), skip_special_tokens=False))
                        print(f"Example ref decoded: {tokenizer.decode(ref_np, skip_special_tokens=False)}")
                        print(f"Example pred decoded: {tokenizer.decode(pred_np, skip_special_tokens=False)}")
                        print(f"Example pred decoded (no TF): {tokenizer.decode(preds_no_tf_np, skip_special_tokens=False)}")
                    except Exception:
                        logger.exception("Failed to decode tokens with tokenizer")

                printed_example = True

def init_params(model: nn.Module, weight_init_method: str = None, bias_init_method: str = None):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if weight_init_method == "xavier_uniform":
                nn.init.xavier_uniform_(param)
            elif weight_init_method == "xavier_normal":
                nn.init.xavier_normal_(param)
        if 'bias' in name and bias_init_method is not None:
            if bias_init_method == "zeros":
                nn.init.zeros_(param)
            elif bias_init_method == "ones":
                nn.init.ones_(param)
    # pass

def log_gradients(model: nn.Module, step_info: dict, writer: SummaryWriter):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), step_info["global_step"])
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, step_info["global_step"])

def save_configs(args, curriculum_levels=None):
    os.makedirs("configs", exist_ok=True)
    model_name = args.name

    json_args = vars(args)
    json_args['curriculum_levels'] = curriculum_levels
    with open(f'configs/{model_name}_config.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_args, indent=4, ensure_ascii=False))

def log_lr(optimizer: torch.optim.Optimizer, step_info: dict, writer: SummaryWriter):
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], step_info["global_step"])

def log_teacher_forcing_ratio(scheduler_sampling: LinearSchedulerSampling, step_info: dict, writer: SummaryWriter):
    writer.add_scalar("teacher_forcing_ratio", scheduler_sampling.get_ratio(), step_info["global_step"])

def log_max_len(sampler: DynamicBatchSampler, step_info: dict, writer: SummaryWriter):
    if sampler is not None:
        max_len = sampler.get_max_length()
        writer.add_scalar("max_len", max_len, step_info["global_step"])

def log_batch_size(sampler: DynamicBatchSampler, step_info: dict, writer: SummaryWriter):
    if sampler is not None:
        batch_size = sampler.get_batch_size()
        writer.add_scalar("batch_size", batch_size, step_info["global_step"])

def train(
    model: LSTM, 
    dataloader: DataLoader,
    dataloader_eval: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    step_info: dict,
    writer: SummaryWriter,
    early_stopper: EarlyStopper,
    scheduler: WarmupScheduler,
    tokenizer: Tokenizer = None,
    scheduler_sampling: LinearSchedulerSampling = None,
    sampler: DynamicBatchSampler = None,
    sampler_eval: DynamicBatchSampler = None
):
    stop = False
    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)
    for _ in range(int(1e6)):
        # for src, tgt in tqdm(dataloader, desc="Training", total=len(dataloader)):
        for src, tgt in dataloader:

            src: torch.Tensor
            tgt: torch.Tensor
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])
            if tokenizer:
                logger.debug(f'Src tokens: {src[0]}')
                logger.debug(f'Src: {tokenizer.decode(src[0].cpu().numpy(), False)}')
                logger.debug(f'Tgt tokens: {tgt[0, :-1]}')
                logger.debug(f'Tgt: {tokenizer.decode(tgt[0].cpu().numpy(), False)}')
                logger.debug(f'Tgt tokens shifted: {tgt[0, 1:]}')
                logger.debug(f'Tgt shifted: {tokenizer.decode(tgt[0, 1:].cpu().numpy(), False)}')

            loss: torch.Tensor = model.train_step(
                src, 
                tgt, 
                criterion, 
                train_config["teacher_forcing"], 
                scheduler_sampling=scheduler_sampling)
            loss = loss / sampler.get_accum_steps()
            loss.backward()
            step_info["loss_train"] += loss.item()
            step_info["loss_train_count"] += 1
            step_info["step"] += 1

            if train_config["clip_grad"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config["clip_grad"])
            if step_info["step"] % sampler.get_accum_steps() == 0:
                optimizer.step()
                scheduler.step()
                scheduler_sampling.step()
                sampler.step()
                sampler_eval.step()

                step_info["global_step"] += 1
                
                if step_info["global_step"] % train_config["log_steps"] == 0:
                    avg_loss = step_info["loss_train"] / step_info["loss_train_count"]
                    logger.debug(f'Step {step_info["global_step"]} - Loss: {avg_loss:.4f}')
                    writer.add_scalar("Loss/Train", avg_loss, step_info["global_step"])
                    log_gradients(model, step_info, writer)
                    step_info["loss_train"] = 0
                    step_info["loss_train_count"] = 0

                    log_lr(optimizer, step_info, writer)
                    log_teacher_forcing_ratio(scheduler_sampling, step_info, writer)
                    log_max_len(sampler, step_info, writer)
                    log_batch_size(sampler, step_info, writer)

                optimizer.zero_grad()

                if step_info["global_step"] % train_config["save_steps"] == 0:
                    os.makedirs("artifacts", exist_ok=True)
                    logger.debug(f'Model saved at step {step_info["global_step"]}')

                if step_info["global_step"] % train_config["eval_steps"] == 0:
                    eval(model, dataloader_eval, criterion, step_info, writer, train_config, tokenizer)
                    eval_loss = step_info["loss_eval"] / len(dataloader_eval)
                    eval_loss_no_tf = step_info["loss_no_tf_eval"] / len(dataloader_eval)
                    logger.debug(f'Step {step_info["global_step"]} - Eval Loss: {eval_loss:.4f}')
                    writer.add_scalar("Loss/Eval", eval_loss, step_info["global_step"])
                    writer.add_scalar("Loss/Eval_no_teacher_forcing", eval_loss_no_tf, step_info["global_step"])
                    step_info["loss_eval"] = 0
                    step_info["loss_no_tf_eval"] = 0

                    if eval_loss < step_info["best_eval_loss"]:
                        step_info["best_eval_loss"] = eval_loss
                        torch.save(model.state_dict(), f'artifacts/model_{args.name}.pt')
                        logger.debug(f'New best model saved at step {step_info["global_step"]} with eval loss {eval_loss:.4f}')

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
    args.add_argument("--vocab_size", type=int, default=50000)
    args.add_argument("--max_len", type=int, default=60)
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

    logger.info(f'Starting training - {args.desc}')
    df_train = pd.read_parquet("data/tokenized_train.parquet")#.sample(n=256, random_state=42).reset_index(drop=True)
    df_train = df_train.loc[df_train[f'en_tokens_{args.vocab_size}_len'] <= args.max_len].reset_index(drop=True)
    # df_train.to_parquet("data/tokenized_train_amostrado.parquet", index=False)
    df_eval = pd.read_parquet("data/tokenized_eval.parquet")
    df_eval = df_eval.loc[df_eval[f'en_tokens_{args.vocab_size}_len'] <= args.max_len].reset_index(drop=True).sample(frac=0.15, random_state=42)

    logger.info("Creating dataset...")
    dataset = TranslateDataset(
        tokens_src=df_train[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_train[f'pt_tokens_{args.vocab_size}'].tolist(),
        invert_src=args.invert_src,
        max_len=int(args.max_len*1.5)
    )

    curriculum_levels = [
        # {"max_step": 15000, "max_len": 10, "batch_size": 384, "accum_steps": 1},
        {"max_step": args.max_steps, "max_len": args.max_len, "batch_size": args.batch_size, "accum_steps": args.accum_steps},
    ]


    sampler = CurriculumLengthSampler(
        tokens_src=df_train[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_train[f'pt_tokens_{args.vocab_size}'].tolist(),
        len_tokens=args.max_len,
        curriculum_levels=curriculum_levels
    )
    batch_sampler = DynamicBatchSampler(sampler)
    collator = DynamicCollator(batch_sampler)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collator)

    tokenizer = Tokenizer.from_file(f'artifacts/tokenizer_{args.vocab_size}.json')

    dataset_eval = TranslateDataset(
        tokens_src=df_eval[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_eval[f'pt_tokens_{args.vocab_size}'].tolist(),
        invert_src=args.invert_src,
        max_len=int(args.max_len*1.5)
    )

    sampler_eval = CurriculumLengthSampler(
        tokens_src=df_eval[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_eval[f'pt_tokens_{args.vocab_size}'].tolist(),
        len_tokens=args.max_len,
        curriculum_levels=curriculum_levels
    )
    batch_sampler_eval = DynamicBatchSampler(sampler_eval)
    collator_eval = DynamicCollator(batch_sampler_eval)
    dataloader_eval = DataLoader(dataset_eval, batch_sampler=batch_sampler_eval, collate_fn=collator_eval)

    if args.architecture == "transformer":
        
        logger.info("Creating model...")
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
            separate_embedding=args.separate_embedding
        )
    else:
        logger.info("Creating model...")
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
            attention=args.attention
        )
    model = model.to(args.device)

    init_params(model, weight_init_method=args.init_weight_method, bias_init_method=args.init_bias_method)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)
    scheduler_sampling = SigmoidSchedulerSampling(
        teacher_forcing_ratio=args.teacher_forcing_ratio, 
        max_steps=args.max_steps_scheduler_sampling, 
        use=args.scheduler_sampling
    )

    train_config = {
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "device": args.device,
        "max_steps": args.max_steps,
        "clip_grad": args.clip_grad,
        "teacher_forcing": args.teacher_forcing
    }

    step_info = {
        "loss_train": 0,
        "loss_train_count": 0,
        "loss_eval": 0,
        "loss_no_tf_eval": 0,
        "step": 0,
        'best_eval_loss': float('inf'),
        'global_step': 0
    }

    writer = SummaryWriter(log_dir=f"runs/{args.name}")
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)

    save_configs(args, curriculum_levels=curriculum_levels)

    logger.info("Starting training loop...")
    train(
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
        tokenizer,
        scheduler_sampling,
        batch_sampler,
        batch_sampler_eval
    )








## python training.py --invert_src --embedding_dim 256 --encoder_hidden_dim 512 --decoder_hidden_dim 512 --encoder_num_layers 2 --decoder_num_layers 2 --encoder_dropout 0.1 --decoder_dropout 0.1 --encoder_bidirectional --batch_size 64 --accum_steps 1 --log_steps 500 --save_steps 5000 --eval_steps 2500 --device cuda --desc "Treinamento inicial" --name "v1"