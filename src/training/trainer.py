"""
CNN Model Trainer – IDS Version (Binary & Multiclass Support)

Features:
✓ Train / Validation loops
✓ Epoch-level timing (train & val)
✓ Early stopping
✓ LR scheduler support
✓ Checkpoint saving
✓ Training history storage
✓ Final test evaluation with latency metrics
"""

import os
import time
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.training.metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    get_predictions_from_logits,
)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        scheduler: Optional[object] = None,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "lr": [],
            "train_time": [],
            "val_time": [],
        }

        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    # =====================================================
    # TRAIN ONE EPOCH
    # =====================================================
    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        start_time = time.time()

        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = get_predictions_from_logits(outputs)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        return {
            "loss": running_loss / len(train_loader),
            "acc": 100.0 * correct / total,
            "time": time.time() - start_time,
        }
    

    # =====================================================
    # VALIDATION
    # =====================================================
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        start_time = time.time()

        running_loss = 0.0
        preds_all, targets_all = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                preds = get_predictions_from_logits(outputs)

                preds_all.append(preds.cpu())
                targets_all.append(targets.cpu())

        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)

        return {
            "loss": running_loss / len(val_loader),
            "acc": accuracy(preds_all, targets_all),
            "precision": precision(preds_all, targets_all),
            "recall": recall(preds_all, targets_all),
            "f1_score": f1_score(preds_all, targets_all),
            "time": time.time() - start_time,
        }

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ):
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_stats = self.train_epoch(train_loader)
            val_stats = self.validate(val_loader)

            # Save history
            self.history["train_loss"].append(train_stats["loss"])
            self.history["train_acc"].append(train_stats["acc"])
            self.history["val_loss"].append(val_stats["loss"])
            self.history["val_acc"].append(val_stats["acc"])
            self.history["precision"].append(val_stats["precision"])
            self.history["recall"].append(val_stats["recall"])
            self.history["f1_score"].append(val_stats["f1_score"])
            self.history["train_time"].append(train_stats["time"])
            self.history["val_time"].append(val_stats["time"])

            # Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_stats["loss"])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["lr"].append(current_lr)

            # Logging
            if verbose:
                print(f"\nEpoch [{epoch}/{epochs}]")
                print(f"  Train Loss : {train_stats['loss']:.4f} | Train Acc : {train_stats['acc']:.2f}%")
                print(f"  Val Loss   : {val_stats['loss']:.4f} | Val Acc   : {val_stats['acc']:.2f}%")
                print(f"  Precision  : {val_stats['precision']:.4f}")
                print(f"  Recall     : {val_stats['recall']:.4f}")
                print(f"  F1 Score   : {val_stats['f1_score']:.4f}")
                print(f"  Train Time : {train_stats['time']:.2f}s | Val Time : {val_stats['time']:.2f}s")
                print(f"  LR         : {current_lr:.6f}")

            # Early stopping
            if val_stats["loss"] < self.best_val_loss:
                self.best_val_loss = val_stats["loss"]
                self.best_val_acc = val_stats["acc"]
                patience_counter = 0
                best_epoch = epoch

                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch} (best epoch {best_epoch})")
                    break

        return self.history

    # =====================================================
    # FINAL TEST (LATENCY BENCHMARK)
    # =====================================================
    def test(self, test_loader: DataLoader):
        self.model.eval()
        start_time = time.time()

        preds_all, targets_all = [], []
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                preds = get_predictions_from_logits(outputs)
                preds_all.append(preds.cpu())
                targets_all.append(targets)

                total_samples += targets.size(0)

        total_time = time.time() - start_time
        time_per_sample = total_time / total_samples

        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)

        results = {
            "accuracy": float(accuracy(preds_all, targets_all)),
            "precision": float(precision(preds_all, targets_all)),
            "recall": float(recall(preds_all, targets_all)),
            "f1_score": float(f1_score(preds_all, targets_all)),
            "test_time_sec": total_time,
            "samples": total_samples,
            "time_per_sample_ms": time_per_sample * 1000,
            "samples_per_sec": total_samples / total_time,
        }

        print("\n" + "=" * 40)
        print("FINAL TEST RESULTS")
        print("=" * 40)
        for k, v in results.items():
            print(f"{k:20s}: {v}")

        return results
