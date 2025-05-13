import os
from typing import Optional

import numpy as np
import torch
import typer
from typing_extensions import Annotated

from models import VIB
from train import test, train
from utils import (
    load_checkpoint,
    load_mnist,
    plot_metrics,
    save_checkpoint,
    visualize_latent_space,
)

app = typer.Typer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.command()
def main(
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 50,
    batch_size: Annotated[int, typer.Option(help="Batch size for training")] = 128,
    latent_dim: Annotated[int, typer.Option(help="Dimension of latent space")] = 256,
    hidden_dim: Annotated[int, typer.Option(help="Dimension of hidden layers")] = 1024,
    learning_rate: Annotated[float, typer.Option(help="Learning rate")] = 1e-3,
    beta: Annotated[float, typer.Option(help="Weight for KL term in VIB loss")] = 1e-3,
    output_dir: Annotated[
        str, typer.Option(help="Directory to save results")
    ] = "results",
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    evaluate_only: Annotated[
        bool, typer.Option(help="Only evaluate without training")
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Resume training from checkpoint")
    ] = False,
    checkpoint_dir: Annotated[
        str, typer.Option(help="Directory to save/load checkpoints")
    ] = "checkpoints",
):
    """
    Main function to train, evaluate, and visualize the Deep Variational Information Bottleneck model.
    """
    try:
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load MNIST dataset
        train_loader, test_loader = load_mnist(batch_size)

        # Initialize model and optimizer
        model = VIB(
            input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim, output_dim=10
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize metrics trackers
        train_metrics = {
            "total_loss": [],
            "ce_loss": [],
            "kl_loss": [],
            "train_acc": [],
        }

        test_metrics = {"total_loss": [], "ce_loss": [], "kl_loss": [], "test_acc": []}

        # Track best validation loss
        best_test_loss = float("inf")
        start_epoch = 1

        # Load checkpoint if resuming
        if resume or evaluate_only:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                start_epoch, loaded_metrics = load_checkpoint(
                    model, optimizer, checkpoint_path
                )
                best_test_loss = (
                    min(test_metrics["total_loss"])
                    if test_metrics["total_loss"]
                    else float("inf")
                )
                print(f"Resuming training from epoch {start_epoch}")
            else:
                print("No checkpoint found, starting from scratch")
                if evaluate_only:
                    raise FileNotFoundError("No checkpoint found for evaluation")

        # Evaluation only mode
        if evaluate_only:
            print("Running evaluation only")
            test_epoch_metrics = test(model, test_loader, beta, device)

            print("\nEvaluation Results:")
            print(
                f"Test Loss: {test_epoch_metrics['total_loss']:.4f} "
                f"(CE: {test_epoch_metrics['ce_loss']:.4f}, "
                f"KL: {test_epoch_metrics['kl_loss']:.4f}) "
                f"Acc: {test_epoch_metrics['test_acc']:.2f}%"
            )

            # Visualize latent space
            latent_viz_path = os.path.join(output_dir, "latent_space.png")
            visualize_latent_space(model, test_loader, device, latent_viz_path)
            return

        # Training loop
        for epoch in range(start_epoch, epochs + 1):
            # Train
            train_epoch_metrics = train(
                model, train_loader, optimizer, epoch, beta, device
            )

            # Test
            test_epoch_metrics = test(model, test_loader, beta, device)

            # Store metrics
            for k in train_metrics:
                if k in train_epoch_metrics:
                    train_metrics[k].append(train_epoch_metrics[k])

            for k in test_metrics:
                if k in test_epoch_metrics:
                    test_metrics[k].append(test_epoch_metrics[k])

            # Checkpointing
            is_best = test_epoch_metrics["total_loss"] < best_test_loss
            if is_best:
                best_test_loss = test_epoch_metrics["total_loss"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=test_epoch_metrics,
                is_best=is_best,
                checkpoint_dir=checkpoint_dir,
            )

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(
                f"Train Loss: {train_epoch_metrics['total_loss']:.4f} "
                f"(CE: {train_epoch_metrics['ce_loss']:.4f}, "
                f"KL: {train_epoch_metrics['kl_loss']:.4f}) "
                f"Acc: {train_epoch_metrics['train_acc']:.2f}%"
            )

            print(
                f"Test Loss: {test_epoch_metrics['total_loss']:.4f} "
                f"(CE: {test_epoch_metrics['ce_loss']:.4f}, "
                f"KL: {test_epoch_metrics['kl_loss']:.4f}) "
                f"Acc: {test_epoch_metrics['test_acc']:.2f}%"
            )

        # Plot and save metrics
        plot_metrics(train_metrics, test_metrics, output_dir)

        # Visualize latent space
        latent_viz_path = os.path.join(output_dir, "latent_space.png")
        visualize_latent_space(model, test_loader, device, latent_viz_path)

        # Save final model
        model_path = os.path.join(output_dir, "vib_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nTraining complete. Model saved to {model_path}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    app()
