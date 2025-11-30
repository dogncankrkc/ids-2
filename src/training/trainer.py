"""
CNN Model Trainer

This module provides a Trainer class for training and evaluating
CNN models with various features like learning rate scheduling,
early stopping, and checkpoint saving.
"""

import os
import time
from typing import Dict, Optional, Callable, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class Trainer:
    """
    Trainer class for CNN model training and evaluation.

    Features:
        - Training and validation loops
        - Learning rate scheduling
        - Early stopping
        - Model checkpointing
        - Training history tracking

    Args:
        model: PyTorch model to train
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to train on (cuda/cpu)
        scheduler: Optional learning rate scheduler
        checkpoint_dir: Directory to save model checkpoints
    """

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
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Best model tracking
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total

        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Whether to print training progress

        Returns:
            Training history dictionary
        """
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validation
            if val_loader:
                val_metrics = self.validate(val_loader)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

                # Check for improvement
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.best_val_acc = val_metrics["accuracy"]
                    self.save_checkpoint("best_model.pth", epoch, val_metrics)
                    patience_counter = 0
                else:
                    patience_counter += 1

            # Learning rate scheduling
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_loader:
                        self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Logging
            if verbose:
                elapsed = time.time() - start_time
                msg = f"Epoch {epoch+1}/{epochs} | "
                msg += f"Train Loss: {train_metrics['loss']:.4f}, "
                msg += f"Train Acc: {train_metrics['accuracy']:.2f}%"

                if val_loader:
                    msg += f" | Val Loss: {val_metrics['loss']:.4f}, "
                    msg += f"Val Acc: {val_metrics['accuracy']:.2f}%"

                msg += f" | LR: {current_lr:.6f} | Time: {elapsed:.2f}s"
                print(msg)

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch)

        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            filename: Name of the checkpoint file
            epoch: Current epoch number
            metrics: Optional metrics to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "metrics": metrics or {},
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            filename: Name of the checkpoint file

        Returns:
            Checkpoint dictionary
        """
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.history = checkpoint.get("history", self.history)

        return checkpoint
