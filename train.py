from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import loss_function


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    beta: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    metrics_accum = {"total_loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0}
    correct, total = 0, 0

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for data, target in tepoch:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            pred, mean, logvar = model(data)
            loss, metrics = loss_function(pred, target, mean, logvar, beta)
            loss.backward()
            optimizer.step()

            for k in metrics_accum:
                if k in metrics:
                    metrics_accum[k] += metrics[k]

            _, predicted = pred.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            tepoch.set_postfix(
                {
                    "loss": metrics["total_loss"],
                    "ce_loss": metrics["ce_loss"],
                    "kl_loss": metrics["kl_loss"],
                    "acc": 100.0 * correct / total,
                }
            )

    for k in metrics_accum:
        metrics_accum[k] /= len(train_loader)
    metrics_accum["train_acc"] = 100.0 * correct / total
    return metrics_accum


def test(
    model: nn.Module, test_loader: DataLoader, beta: float, device: torch.device
) -> Dict[str, float]:
    """Evaluate the model on test data."""
    model.eval()
    metrics_accum = {"total_loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0}
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred, mean, logvar = model(data)
            _, metrics = loss_function(pred, target, mean, logvar, beta)

            for k in metrics_accum:
                if k in metrics:
                    metrics_accum[k] += metrics[k]

            _, predicted = pred.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

    for k in metrics_accum:
        metrics_accum[k] /= len(test_loader)
    metrics_accum["test_acc"] = 100.0 * correct / total
    return metrics_accum
