import os
import math
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset and create data loaders."""
    try:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten the image
            ]
        )

        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )
        return train_loader, test_loader

    except Exception as e:
        print(f"Error loading MNIST dataset: {e}")
        raise


def loss_function(
    pred: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    beta: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculate VIB loss (cross entropy + KL divergence)."""
    ce_loss = F.cross_entropy(pred, target) / math.log(2)
    kl_loss = -0.5 * (1 + std - mean.pow(2) - std.exp()).sum(1).mean() / math.log(2)
    total_loss = ce_loss + beta * kl_loss

    metrics = {
        "total_loss": total_loss.item(),
        "ce_loss": ce_loss.item(),
        "kl_loss": kl_loss.item(),
        "beta": beta,
    }

    return total_loss, metrics


def visualize_latent_space(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: str,
    n_samples: int = 1000,
    random_state: int = 42,
) -> None:
    """
    Visualize the latent space using t-SNE and save as PNG.

    Args:
        model: Trained VIB model
        data_loader: DataLoader for the data to visualize
        device: Device to run inference on
        output_path: Path to save the visualization
        n_samples: Number of samples to visualize
        random_state: Random seed for t-SNE
    """
    try:
        model.eval()
        latent_vectors = []
        labels = []

        # Collect latent vectors and labels
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(device)
                z = model.encode(data).cpu().numpy()
                latent_vectors.append(z)
                labels.append(target.cpu().numpy())

                # Stop if we have enough samples
                if sum(len(x) for x in latent_vectors) >= n_samples:
                    break

        latent_vectors = np.concatenate(latent_vectors)[:n_samples]
        labels = np.concatenate(labels)[:n_samples]

        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, random_state=random_state)
        latent_2d = tsne.fit_transform(latent_vectors)

        # Create plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab10", alpha=0.6
        )
        plt.colorbar(scatter, ticks=range(10))
        plt.title("Latent Space Visualization (t-SNE)")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")

        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Latent space visualization saved to {output_path}")

    except Exception as e:
        print(f"Error visualizing latent space: {e}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
    """
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save regular checkpoint
        torch.save(state, os.path.join(checkpoint_dir, "checkpoint.pth"))

        # Save best checkpoint separately
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, "best_model.pth"))
            print(
                f"New best model saved at epoch {epoch} with test loss {metrics['total_loss']:.4f}"
            )

    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str
) -> Tuple[int, Dict[str, Any]]:
    """
    Load model checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (epoch, metrics) from the checkpoint
    """
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"], checkpoint["metrics"]

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def plot_metrics(
    train_metrics: Dict[str, list],
    test_metrics: Dict[str, list],
    output_dir: str = "results",
) -> None:
    """Plot training and testing metrics."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        epochs = range(1, len(train_metrics["total_loss"]) + 1)

        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_metrics["total_loss"], label="Train Loss")
        plt.plot(epochs, test_metrics["total_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

        # Plot accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_metrics["train_acc"], label="Train Accuracy")
        plt.plot(epochs, test_metrics["test_acc"], label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Classification Accuracy")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "accuracy.png"))
        plt.close()

        # Plot KL divergence
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_metrics["kl_loss"], label="Train KL Divergence")
        plt.plot(epochs, test_metrics["kl_loss"], label="Test KL Divergence")
        plt.xlabel("Epoch")
        plt.ylabel("KL Loss")
        plt.title("KL Divergence")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "kl_divergence.png"))
        plt.close()

    except Exception as e:
        print(f"Error plotting metrics: {e}")
