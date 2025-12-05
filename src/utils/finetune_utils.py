from stores import ModelProvidersFactory, DatasetProvidersFactory
from . import train_step, val_step, save_checkpoint, setup_logger

import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv


def get_optimizer(logger, model, lr):
    """Creates the optimizer for training.

    Args:
        model (nn.Module): Model to optimize.
        config (dict): Configuration dictionary.
        verbose (bool): If True, prints info logs.

    Returns:
        torch.optim.Optimizer: AdamW optimizer.
    """
    logger.info("Creating optimizer...")
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    logger.info("Optimizer created: AdamW Optimizer.")
    logger.info(f"Initial Learning Rate: {lr}.")
    return optimizer


def get_scheduler(logger, optimizer):
    logger.info("Creating Scheduler...")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    logger.info("Scheduler Created Successfully!")
    return scheduler

def set_all_seeds(logger, seed_value: int) -> None:
        """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

        Args:
            seed_value (int): The seed value to set.
        """

        logger.info("Setting all seeds...")
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ["PYTHONHASHSEED"] = str(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def finetune(log_dir: str,
             lr,
             num_epochs: int,
             batch_size: int,
             videos_root: str,
             annot_root: str,
             train_ids: list[int],
             val_ids: list[int],
             features: bool,
             model_name: str,
             num_classes: int,
             actions_dict: dict,
             metrics_logs: str,
             preds_logs: str,
             save_path: str,
             use_scheduler: bool = True,
             image_level: bool = None,
             crop: bool = None,
             output_file: str = None,
             input_size: int = None,
             hidden_size1: int = None,
             hidden_size2: int = None,
             num_layers: int = None,
             sequence: bool = None,
             verbose: bool = False):
    """Main function to run training and testing loop.

    Args:
        verbose (bool): If True, prints info logs.
    """
    logger = setup_logger(
            log_file=__file__,
            log_dir=log_dir,
            log_to_console=verbose,
            use_tqdm=True,
        )
    
    set_all_seeds(logger, 42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_factory = DatasetProvidersFactory()
    trainloader, valloader = dataset_factory.get_data_loaders(
                                                batch_size,
                                                videos_root,
                                                annot_root,
                                                train_ids,
                                                val_ids,
                                                features,
                                                log_dir,
                                                actions_dict,
                                                output_file,
                                                image_level,
                                                crop,
                                                sequence,
                                                verbose
                                            )
    
    models_factory = ModelProvidersFactory()
    model = models_factory.create(model_name=model_name, num_classes=num_classes,
                                         input_size=input_size, hidden_size1=hidden_size1,
                                         hidden_size2=hidden_size2, num_layers=num_layers,
                                         log_dir=log_dir, verbose=verbose)
    
    criterion = nn.CrossEntropyLoss()
    optimizer= get_optimizer(logger, model, lr)
    scheduler = get_scheduler(logger, optimizer) if use_scheduler else None

    logger.info(f"Starting Training and Testing with number of epochs = {num_epochs}")

    # Create or open a CSV file and define headers
    logger.info("Opening a CSV file to log training metrics.")
    with open(metrics_logs, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "test_f1"])

    
    Y_true = []
    Y_pred = []
    best_acc = 0
    for epoch in range(num_epochs):

        logger.info(f"Epoch {epoch+1}\n-----------------------")

        # Training step
        logger.info("Starting training step...")
        model, train_loss, train_acc = train_step(
            data_loader=trainloader,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            device=device,
            logs_path=log_dir,
            num_classes=num_classes,
            verbose=verbose
        )
        logger.info("Training step completed.")

        # Validation step
        logger.info("Starting Validation step...")
        val_f1_score, val_loss, val_acc, y_true, y_pred = val_step(
            data_loader=valloader,
            model=model,
            loss_fn=criterion,
            device=device,
            logs_path=log_dir,
            num_classes=num_classes,
            verbose=verbose
        )
        logger.info("Validation step completed.")

        if use_scheduler:
            scheduler.step(val_loss)
        
        Y_true.extend(y_true)
        Y_pred.extend(y_pred)
        
        # Save ONLY ONE checkpoint (best so far)
        if val_acc > best_acc:
            best_acc = val_acc
            logger.info(f"New BEST accuracy: {best_acc:.4f}. Saving best model...")

            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
                'acc': train_acc,
                'val_acc': val_acc
            }

            # Always SAVE OVER the same file
            best_path = os.path.join(save_path, "best_model.pth")
            save_checkpoint(checkpoint, best_path)

            logger.info("Best model saved successfully!")
        
        # Append results to CSV
        logger.info("Appending the epoch's metrics to CSV.")
        with open(metrics_logs, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, val_f1_score.item()])
        logger.info("Epoch's metrics appended successfully!")

    # Save Y_true and Y_pred to CSV (to visualize confusion matrix later)
    df = pd.DataFrame({
        "y_true": Y_true,
        "y_pred": Y_pred
    })
    df.to_csv(preds_logs)

