"""
Training Pipeline for SignBridge Lightweight Temporal Transformer.
Trains on the Federated ISL ConcatDataset combining 5 dataset streams.
Includes AdamW optimizer, CrossEntropyLoss, CosineAnnealingLR scheduler,
and TensorBoard logging.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        """Fallback logger when TensorBoard is not installed."""
        def __init__(self, log_dir=None):
            self.log_dir = log_dir
            print(f"[SummaryWriter] TensorBoard not found. Logging fallback active (log_dir={log_dir}).")

        def add_scalar(self, tag, scalar_value, global_step=None):
            pass

        def close(self):
            pass

# Ensure local modules are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.datasets.dataset import FederatedISLDataset, create_federated_dataloaders, MasterGlossMap
from training.models.model import TemporalTransformer


def set_seed(seed: int = 42):
    """Sets random seeds for reproducible training runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> List[float]:
    """Calculates top-k accuracy for the given predictions and targets."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Handle topk when maxk exceeds number of classes
        if output.size(1) < maxk:
            maxk = output.size(1)
            topk = [k for k in topk if k <= maxk]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    num_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()

        # Gradient clipping for transformer stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc = calculate_accuracy(logits, batch_y, topk=(1,))[0]
        total_loss += loss.item()
        total_top1 += acc
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_top1 = total_top1 / max(num_batches, 1)
    return avg_loss, avg_top1


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """Evaluates the model on validation/test set."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            accs = calculate_accuracy(logits, batch_y, topk=(1, 5))
            top1 = accs[0]
            top5 = accs[1] if len(accs) > 1 else accs[0]

            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_top1 = total_top1 / max(num_batches, 1)
    avg_top5 = total_top5 / max(num_batches, 1)
    return avg_loss, avg_top1, avg_top5


def parse_args():
    parser = argparse.ArgumentParser(description="SignBridge Federated ISL Training Pipeline")
    
    # Dataset path arguments
    parser.add_argument("--include_dir", type=str, default="", help="Path to raw INCLUDE dataset")
    parser.add_argument("--cislr_dir", type=str, default="", help="Path to raw CISLR dataset videos")
    parser.add_argument("--cislr_csv", type=str, default="", help="Path to CISLR dataset.csv")
    parser.add_argument("--mendeley_dir", type=str, default="", help="Path to Mendeley ISL dataset clips")
    parser.add_argument("--kaggle_dir", type=str, default="", help="Path to Kaggle ISL dataset clips")
    parser.add_argument("--bridgeconn_shards", nargs="*", default=[], help="WebDataset tar shard paths")
    parser.add_argument("--custom_json_map", type=str, default="", help="Path to custom JSON gloss alignment map")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer hidden dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer encoder layers")
    parser.add_argument("--nhead", type=int, default=8, help="Multi-head attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence frame length")
    
    # Logging & Checkpoints
    parser.add_argument("--output_dir", type=str, default="./experiments/checkpoints", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="./experiments/logs", help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Select hardware accelerator (Metal MPS for Mac, CUDA for NVIDIA, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"🚀 Initializing SignBridge Training Pipeline on Device: {device}")

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 1. Instantiate Federated Dataset
    fed_dataset = FederatedISLDataset(
        bridgeconn_shards=args.bridgeconn_shards,
        include_dir=args.include_dir,
        cislr_dir=args.cislr_dir,
        cislr_csv=args.cislr_csv,
        mendeley_dir=args.mendeley_dir,
        kaggle_dir=args.kaggle_dir,
        custom_json_map=args.custom_json_map if args.custom_json_map else None,
        target_len=args.seq_len
    )
    fed_dataset.summary()

    # Save MasterGlossMap
    master_gloss_map = fed_dataset.get_gloss_map()
    gloss_map_path = os.path.join(args.output_dir, "master_gloss_map.json")
    master_gloss_map.save(gloss_map_path)
    print(f"Saved Master Gloss Vocabulary Map ({len(master_gloss_map)} classes) to {gloss_map_path}")

    # 2. Build DataLoaders
    train_loader, val_loader, _ = create_federated_dataloaders(
        fed_dataset,
        batch_size=args.batch_size,
        val_split=0.2
    )

    # 3. Instantiate Lightweight Temporal Transformer
    num_classes = max(len(master_gloss_map), 1)
    model = TemporalTransformer(
        input_dim=258,
        num_classes=num_classes,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    # 4. Criterion, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"\nModel Architecture Loaded. Number of Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Starting Training Loop...\n" + "=" * 50)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - start_time

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train_Top1", train_acc, epoch)
        writer.add_scalar("Accuracy/Val_Top1", val_top1, epoch)
        writer.add_scalar("Accuracy/Val_Top5", val_top5, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Top1: {val_top1:.2f}% | Val Top5: {val_top5:.2f}% | LR: {current_lr:.6f}"
        )

        # Save Best Model Checkpoint
        if val_top1 >= best_val_acc:
            best_val_acc = val_top1
            checkpoint_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_top1,
                'num_classes': num_classes,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ⭐ Saved Best Model Checkpoint to {checkpoint_path} (Val Acc: {val_top1:.2f}%)")

    writer.close()
    print("=" * 50 + "\n🎉 Training Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
