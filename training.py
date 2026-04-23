from argparse import ArgumentParser
from datetime import datetime
from torch.utils.data import DataLoader
import logging
import os
from models.lstm import LSTM
from utils.dataset import TranslateDataset
import pandas as pd
from logging import getLogger
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

logger = getLogger(__name__)
logger.setLevel(logging.INFO)

def eval(model: nn.Module, dataloader: DataLoader, criterion: nn.CrossEntropyLoss, step_info: dict, writer: SummaryWriter):
    model.eval()
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
            src = torch.tensor(src).to(train_config["device"])
            tgt = torch.tensor(tgt).to(train_config["device"])

            loss = model.eval_step(src, tgt, criterion)

            step_info["loss_eval"] += loss

def log_gradients(model: nn.Module, step_info: dict, writer: SummaryWriter):
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"gradients/{name}", param.grad.norm().item(), step_info["step"])
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar("gradients/total_norm", total_norm, step_info["step"])

def save_configs(args):
    os.makedirs("configs", exist_ok=True)
    model_name = args.name

    json_args = vars(args)
    with open(f'configs/{model_name}_config.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_args, indent=4, ensure_ascii=False))
        

def train(
    model: nn.Module, 
    dataloader: DataLoader,
    dataloader_eval: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,
    train_config: dict,
    step_info: dict,
    writer: SummaryWriter

):
    for _ in range(int(1e6)):
        for src, tgt in tqdm(dataloader, desc="Training", total=len(dataloader)):
            src: torch.Tensor
            tgt: torch.Tensor
            src = src.to(train_config["device"])
            tgt = tgt.to(train_config["device"])

            tgt = tgt[:, 1:]
            src = src[:, :-1]

            loss = model.train_step(src, tgt, criterion, optimizer)

            step_info["loss_train"] += loss
            step_info["step"] += 1

            if step_info["step"] % train_config["log_steps"] == 0:
                logger.info(f'Step {step_info["step"]} - Loss: {step_info["loss_train"] / train_config["log_steps"]:.4f}')
                writer.add_scalar("Loss/Train", step_info["loss_train"] / train_config["log_steps"], step_info["step"])
                log_gradients(model, step_info, writer)
                step_info["loss_train"] = 0

            if step_info["step"] % train_config["save_steps"] == 0:
                os.makedirs("artifacts", exist_ok=True)
                torch.save(model.state_dict(), f'artifacts/model.pt')
                logger.info(f'Model saved at step {step_info["step"]}')

            if step_info["step"] % train_config["eval_steps"] == 0:
                eval(model, dataloader_eval, criterion, step_info, writer)
                logger.info(f'Step {step_info["step"]} - Eval Loss: {step_info["loss_eval"] / len(dataloader_eval):.4f}')
                writer.add_scalar("Loss/Eval", step_info["loss_eval"] / len(dataloader_eval), step_info["step"])
                step_info["loss_eval"] = 0



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
    args = args.parse_args()

    logger.info(f'Starting training - {args.desc}')
    df_train = pd.read_parquet("data/tokenized_train.parquet")
    df_eval = pd.read_parquet("data/tokenized_eval.parquet")

    logger.info("Creating dataset...")
    dataset = TranslateDataset(
        tokens_src=df_train['en_tokens'].tolist(),
        tokens_tgt=df_train['pt_tokens'].tolist(),
        invert_src=args.invert_src
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    dataset_eval = TranslateDataset(
        tokens_src=df_eval['en_tokens'].tolist(),
        tokens_tgt=df_eval['pt_tokens'].tolist(),
        invert_src=args.invert_src
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
        vocab_size=10000,
        pad_idx=0
    )
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters())

    train_config = {
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "device": args.device
    }

    step_info = {
        "loss_train": 0,
        "loss_eval": 0,
        "step": 0
    }

    writer = SummaryWriter(log_dir=f"runs/{args.name}")

    save_configs(args)

    logger.info("Starting training loop...")
    train(model, dataloader, dataloader_eval, criterion, optimizer, train_config, step_info, writer)








## python training.py --invert_src --embedding_dim 256 --encoder_hidden_dim 512 --decoder_hidden_dim 512 --encoder_num_layers 2 --decoder_num_layers 2 --encoder_dropout 0.1 --decoder_dropout 0.1 --encoder_bidirectional --batch_size 64 --accum_steps 1 --log_steps 500 --save_steps 5000 --eval_steps 2500 --device cuda --desc "Treinamento inicial" --name "v1"