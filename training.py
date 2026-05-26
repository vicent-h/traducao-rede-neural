from argparse import ArgumentParser
from datetime import datetime
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import logging
import os
from models.lstm_proj_diff import LSTM
from utils.dataset import TranslateDataset
from utils.earlystopper import EarlyStopper
import pandas as pd
from logging import getLogger
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
from utils.warmup import WarmupScheduler

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

# handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG)
# logger.addHandler(handler)

def eval(model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, step_info: dict, writer: SummaryWriter, train_config: dict):
    model.eval()
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])

            loss = model.eval_step(src, tgt, criterion)

            step_info["loss_eval"] += loss

def init_params(model: nn.Module):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
    # pass

def log_gradients(model: nn.Module, step_info: dict, writer: SummaryWriter):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), step_info["global_step"])
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, step_info["global_step"])

def save_configs(args):
    os.makedirs("configs", exist_ok=True)
    model_name = args.name

    json_args = vars(args)
    with open(f'configs/{model_name}_config.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_args, indent=4, ensure_ascii=False))

def log_lr(optimizer: torch.optim.Optimizer, step_info: dict, writer: SummaryWriter):
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], step_info["global_step"])

        

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
    tokenizer: Tokenizer = None
):
    stop = False
    for _ in range(int(1e6)):
        for src, tgt in tqdm(dataloader, desc="Training", total=len(dataloader)):
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

            loss: torch.Tensor = model.train_step(src, tgt, criterion, optimizer)
            loss = loss / train_config["accum_steps"]
            loss.backward()
            step_info["loss_train"] += loss.item()
            step_info["loss_train_count"] += 1
            step_info["step"] += 1

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            if step_info["step"] % train_config['accum_steps'] == 0:
                optimizer.step()
                scheduler.step()

                step_info["global_step"] += 1
                
                if step_info["global_step"] % train_config["log_steps"] == 0:
                    avg_loss = step_info["loss_train"] / step_info["loss_train_count"]
                    logger.debug(f'Step {step_info["global_step"]} - Loss: {avg_loss:.4f}')
                    writer.add_scalar("Loss/Train", avg_loss, step_info["global_step"])
                    log_gradients(model, step_info, writer)
                    step_info["loss_train"] = 0
                    step_info["loss_train_count"] = 0

                    log_lr(optimizer, step_info, writer)

                optimizer.zero_grad()

                if step_info["global_step"] % train_config["save_steps"] == 0:
                    os.makedirs("artifacts", exist_ok=True)
                    logger.debug(f'Model saved at step {step_info["global_step"]}')

                if step_info["global_step"] % train_config["eval_steps"] == 0:
                    eval(model, dataloader_eval, criterion, step_info, writer, train_config)
                    eval_loss = step_info["loss_eval"] / len(dataloader_eval)
                    logger.debug(f'Step {step_info["global_step"]} - Eval Loss: {eval_loss:.4f}')
                    writer.add_scalar("Loss/Eval", eval_loss, step_info["global_step"])
                    step_info["loss_eval"] = 0

                    # if eval_loss < step_info["best_eval_loss"]:
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
    args.add_argument("--embedding_dim", type=int, default=256)
    args.add_argument("--encoder_hidden_dim", type=int, default=512)
    args.add_argument("--decoder_hidden_dim", type=int, default=512)
    args.add_argument("--encoder_num_layers", type=int, default=2)
    args.add_argument("--decoder_num_layers", type=int, default=2)
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
    args.add_argument("--early_stop_patience", type=int, default=50000)
    args.add_argument("--early_stop_min_delta", type=float, default=0.0)
    args.add_argument("--warmup_steps", type=int, default=5000)
    args.add_argument("--max_steps", type=int, default=500000)
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--vocab_size", type=int, default=50000)
    args.add_argument("--max_len", type=int, default=60)
    args = args.parse_args()

    logger.info(f'Starting training - {args.desc}')
    df_train = pd.read_parquet("data/tokenized_train.parquet")
    df_eval = pd.read_parquet("data/tokenized_eval.parquet")

    logger.info("Creating dataset...")
    dataset = TranslateDataset(
        tokens_src=df_train[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_train[f'pt_tokens_{args.vocab_size}'].tolist(),
        invert_src=args.invert_src,
        max_len=args.max_len
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    tokenizer = Tokenizer.from_file(f'artifacts/tokenizer_{args.vocab_size}.json')

    dataset_eval = TranslateDataset(
        tokens_src=df_eval[f'en_tokens_{args.vocab_size}'].tolist(),
        tokens_tgt=df_eval[f'pt_tokens_{args.vocab_size}'].tolist(),
        invert_src=args.invert_src,
        max_len=args.max_len
    )

    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False)

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
        pad_idx=0
    )
    model = model.to(args.device)

    init_params(model)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = WarmupScheduler(optimizer, warmup_steps=args.warmup_steps, max_steps=args.max_steps)

    train_config = {
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "device": args.device,
        "max_steps": args.max_steps
    }

    step_info = {
        "loss_train": 0,
        "loss_train_count": 0,
        "loss_eval": 0,
        "step": 0,
        'best_eval_loss': float('inf'),
        'global_step': 0
    }

    writer = SummaryWriter(log_dir=f"runs/{args.name}")
    early_stopper = EarlyStopper(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)

    save_configs(args)

    logger.info("Starting training loop...")
    train(model, dataloader, dataloader_eval, criterion, optimizer, train_config, step_info, writer, early_stopper, scheduler, tokenizer)








## python training.py --invert_src --embedding_dim 256 --encoder_hidden_dim 512 --decoder_hidden_dim 512 --encoder_num_layers 2 --decoder_num_layers 2 --encoder_dropout 0.1 --decoder_dropout 0.1 --encoder_bidirectional --batch_size 64 --accum_steps 1 --log_steps 500 --save_steps 5000 --eval_steps 2500 --device cuda --desc "Treinamento inicial" --name "v1"