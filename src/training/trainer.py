"""
CNN Model Trainer – IDS Version (Binary & Multiclass Support)

This trainer is compatible with:
    - Binary IDS classification (using BCEWithLogitsLoss)
    - Multiclass IDS classification (using CrossEntropyLoss)
    - Evaluation metrics from src/evaluation/metrics.py

Key Features:
    ✓ Train / Validation loops
    ✓ Early stopping
    ✓ Learning rate scheduler support
    ✓ Checkpoint saving
    ✓ Training history storage
"""

import os
import time
from typing import Dict, Optional, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Use metrics.py functions
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
        scheduler: Optional[_LRScheduler] = None,
        checkpoint_dir: str = "models/checkpoints",
    ):
        self.model = model.to(device)
        self.criterion = criterion         # BCE or CrossEntropy
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "lr": [],
        }

        # Best results tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0


    # -------------------------------------------------------
    # One epoch of training
    # -------------------------------------------------------
    def train_epoch(self, train_loader: DataLoader):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)     # logits
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = get_predictions_from_logits(outputs)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        return {
            "loss": running_loss / len(train_loader),
            "acc":  100.0 * correct / total
        }


    # -------------------------------------------------------
    # Validation + Extra Metrics
    # -------------------------------------------------------
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        running_loss, preds_all, targets_all = 0.0, [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                preds = get_predictions_from_logits(outputs)

                preds_all.append(preds.cpu())
                targets_all.append(targets.cpu())

        # Concatenate all batches
        preds_all = torch.cat(preds_all)
        targets_all = torch.cat(targets_all)

        # IDS Metrics
        val_acc = accuracy(preds_all, targets_all)
        val_prec = precision(preds_all, targets_all)
        val_rec = recall(preds_all, targets_all)
        val_f1 = f1_score(preds_all, targets_all)

        return {
            "loss": running_loss / len(val_loader),
            "acc": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1_score": val_f1,
        }

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            early_stopping_patience: int = 5,
            verbose: bool = True,
        ):
            best_epoch = 0
            patience_counter = 0

            for epoch in range(1, epochs + 1):
                # ----------- TRAIN EPOCH --------------
                train_stats = self.train_epoch(train_loader)

                # ----------- VALIDATION ---------------
                val_stats = self.validate(val_loader)

                # Save metrics
                self.history["train_loss"].append(train_stats["loss"])
                self.history["train_acc"].append(train_stats["acc"])
                self.history["val_loss"].append(val_stats["loss"])
                self.history["val_acc"].append(val_stats["acc"])
                self.history["precision"].append(val_stats["precision"])
                self.history["recall"].append(val_stats["recall"])
                self.history["f1_score"].append(val_stats["f1_score"])

                # Scheduler varsa → step
                if self.scheduler:
                    self.scheduler.step()
                    self.history["lr"].append(self.scheduler.get_last_lr()[0])

                # ----------- PRINT RESULTS ----------
                if verbose:
                    print(f"\nEpoch [{epoch}/{epochs}]")
                    print(f"  Train Loss : {train_stats['loss']:.4f} | Train Acc : {train_stats['acc']:.2f}%")
                    print(f"  Val Loss   : {val_stats['loss']:.4f} | Val Acc   : {val_stats['acc']:.2f}%")
                    print(f"  Precision  : {val_stats['precision']:.4f}")
                    print(f"  Recall     : {val_stats['recall']:.4f}")
                    print(f"  F1 Score   : {val_stats['f1_score']:.4f}")
                    if self.scheduler:
                        print(f"  LR         : {self.history['lr'][-1]:.6f}")

                # ----------- EARLY STOPPING ----------
                if val_stats["loss"] < self.best_val_loss:
                    self.best_val_loss = val_stats["loss"]
                    self.best_val_acc  = val_stats["acc"]
                    patience_counter = 0
                    best_epoch = epoch

                    # SAVING BEST CHECKPOINT
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))

                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered at epoch {epoch} (best was {best_epoch})")
                        break

            return self.history