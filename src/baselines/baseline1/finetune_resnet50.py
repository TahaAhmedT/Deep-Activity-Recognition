from src.baselines.baseline1.dataset import B1Dataset
from src.baselines.baseline1.extended_model import ExtendedModel
from src.utils.train_utils import train_step
from src.utils.test_utils import test_step
from src.utils.checkpoints_utils import save_checkpoint
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logger

import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import csv

logger = setup_logger(
            log_file=__file__,
            log_dir="logs/baselines_logs/baseline1_logs",
            log_to_console=True,
            use_tqdm=True,
        )

def get_data_loaders(config, verbose=False):
    """Creates train and test data loaders.

    Args:
        config (dict): Configuration dictionary.
        verbose (bool): If True, prints info logs.

    Returns:
        tuple: (trainloader, testloader)
    """
    logger.info("Creating data loaders...")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = config["TRAINING_PARAMS"]["batch_size"]
    train_dataset = B1Dataset(
        videos_root=config["DATA_PATHS"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["train_ids"],
        transform=transform
    )
    test_dataset = B1Dataset(
        videos_root=config["DATA_PATHS"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["val_ids"],
        transform=transform
    )
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Data loaders created.")
    return trainloader, testloader

def get_model(logger, verbose=False):
    """Initializes the ResNet50 model and truncates the last layer.

    Args:
        verbose (bool): If True, prints info logs.

    Returns:
        nn.Module: Modified ResNet50 model wrapped in ExtendedModel.
    """
    logger.info("Initializing ResNet50 model...")
    original_model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=verbose)
    layers = list(original_model.children())[:-1]
    truncated_model = nn.Sequential(*layers)
    model = ExtendedModel(truncated_model)
    logger.info("Model initialized and truncated.")
    return model

def get_optimizer(model, config, logger):
    """Creates the optimizer for training.

    Args:
        model (nn.Module): Model to optimize.
        config (dict): Configuration dictionary.
        verbose (bool): If True, prints info logs.

    Returns:
        torch.optim.Optimizer: AdamW optimizer.
    """
    logger.info("Creating optimizer...")
    lr = config["TRAINING_PARAMS"]["lr"]
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min")
    logger.info("Optimizer created: AdamW Optimizer.")
    logger.info("Learning Rate Scheduler Created: ReduceLROnPlateau.")
    logger.info(f"Initial Learning Rate: {lr}.")
    return optimizer, scheduler

def set_all_seeds(seed_value: int, logger) -> None:
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

def main(verbose=True):
    """Main function to run training and testing loop.

    Args:
        verbose (bool): If True, prints info logs.
    """
    set_all_seeds(42, logger=logger)
    CONFIG = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    trainloader, testloader = get_data_loaders(CONFIG, verbose=verbose)
    model = get_model(logger=logger, verbose=verbose)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(model, CONFIG)
    num_epochs = CONFIG["TRAINING_PARAMS"]["num_epochs"]
    logger.info(f"Starting Training and Testing with number of epochs = {num_epochs}")

    # Create or open a CSV file and define headers
    logger.info("Opening a CSV file to log training metrics.")
    with open("logs/training_logs/b1_training.csv", mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_f1"])

    
    Y_true = []
    Y_pred = []
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}\n-----------------------")
        # Training step
        logger.info("Starting training step...")
        model, train_loss, train_acc = train_step(
            data_loader=trainloader,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            device=device
        )
        logger.info("Training step completed.")

        # Saving Checkpoint
        if (epoch + 1) % 2 == 0:
            logger.info("Saving model checkpoint...")
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': train_loss,
                'acc': train_acc
            }
            save_checkpoint(checkpoint, f"models/b1_models/checkpoints/epoch_{epoch}.pth")
            logger.info(f"Model checkpoint saved at epoch {epoch+1}/{num_epochs}.")

        
        # Testing step
        logger.info("Starting testing step...")
        test_f1_score, test_loss, test_acc, y_true, y_pred = test_step(
            data_loader=testloader,
            model=model,
            loss_fn=criterion,
            device=device
        )
        Y_true.extend(y_true)
        Y_pred.extend(y_pred)
        # scheduler.step(test_loss)
        logger.info("Testing step completed.")

        # Append results to CSV
        logger.info("Appending the epoch's metrics to CSV.")
        with open("logs/training_logs/b1_training.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc, test_f1_score.item()])
        logger.info("Epoch's metrics appended successfully!")

    # Save Y_true and Y_pred to CSV (to visualize confusion matrix later)
    df = pd.DataFrame({
        "y_true": Y_true,
        "y_pred": Y_pred
    })
    df.to_csv("logs/training_logs/b1_test_predictions.csv")

if __name__ == "__main__":
    main()