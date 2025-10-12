"""
This script fine-tunes a ResNet-50 model for volleyball activity recognition using cropped player images.
It loads the B3Dataset, applies transformations, trains and evaluates the model, and logs metrics and predictions.
"""

from src.baselines.baseline3.dataset import B3Dataset
from src.baselines.baseline1.extended_model import ExtendedModel
from src.baselines.baseline1.finetune_resnet50 import set_all_seeds, get_model, get_optimizer
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

CONFIG = load_config()

logger = setup_logger(
    log_file=__file__,
    log_dir=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
    log_to_console=True,
    use_tqdm=True
)

def get_data_loaders(config):
    """Creates and returns training and testing data loaders for the B3Dataset.

    Args:
        config (dict): Configuration dictionary containing data paths and parameters.

    Returns:
        tuple: (trainloader, testloader) DataLoader objects for training and testing.
    """
    logger.info("Creating Data Loaders...")
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    batch_size = config["TRAINING_PARAMS"]["batch_size"]
    train_dataset = B3Dataset(
        videos_root=config["DATA_PATHS"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["train_ids"],
        annot_root=config["DATA_PATHS"]["annot_root"],
        transform=transform,
        verbose=config["verbose"]
    )
    test_dataset = B3Dataset(
        videos_root=config["DATA_PATHS"]["videos_root"],
        target_videos=config["TARGET_VIDEOS"]["val_ids"],
        annot_root=config["DATA_PATHS"]["annot_root"],
        transform=transform,
        verbose=config["verbose"]
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Data Loaders Created Successfully!")
    return trainloader, testloader


def main(verbose=True):
    """Main function to fine-tune ResNet-50 on the B3Dataset.

    Sets seeds, prepares data loaders, trains and evaluates the model, and logs metrics and predictions.

    Args:
        verbose (bool, optional): If True, enables verbose logging. Defaults to True.
    """
    set_all_seeds(42, logger=logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    trainloader, testloader = get_data_loaders(CONFIG)
    model = get_model(logger=logger, verbose=CONFIG["verbose"])
    criterion = nn.CrossEntropyLoss()
    optimizer, _ = get_optimizer(model=model, config=CONFIG, logger=logger)
    num_epochs = CONFIG["TRAINING_PARAMS"]["num_epochs"]
    logger.info(f"Starting Training and Testing with number of epochs = {num_epochs}")

    # Create or open a CSV file and define headers
    logger.info("Opening a CSV file to log training metrics.")
    with open(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_training.csv"), mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "train_loss", "train_acc", "test_loss", "test_acc", "test_f1"])
    
    Y_true = []
    Y_pred = []
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}\n------------------------")
        # Training step
        logger.info("Starting Training Step...")
        model, train_loss, train_acc = train_step(
            data_loader=trainloader,
            model=model,
            loss_fn=criterion,
            optimizer=optimizer,
            device=device,
            logs_path=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
            verbose=CONFIG["verbose"]
        )
        logger.info("Training Step Completed.")

        # Save Checkpoint
        if (epoch + 1) % 2 == 0:
            logger.info("Saving model checkpoint...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": train_loss,
                "acc": train_acc
            }
            save_checkpoint(checkpoint, os.path.join(CONFIG["MODELS"], f"b3_models/checkpoints/epoch_{epoch+1}.pth"))
            logger.info(f"Model checkpoint saved at epoch {epoch+1}/{num_epochs}")

        # Testing step
        logger.info("Starting Testing Step...")
        test_f1_score, test_loss, test_acc, y_true, y_pred = test_step(
            data_loader=testloader,
            model=model,
            loss_fn=criterion,
            device=device,
            logs_path=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
            verbose=CONFIG["verbose"]
        )
        Y_true.extend(y_true)
        Y_pred.extend(y_pred)
        logger.info("Testing step completed.")

        # Append results to CSV
        logger.info("Appending the epoch's metrics to CSV.")
        with open(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_training.csv"), mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc, test_f1_score.item()])
        logger.info("Epoch's metrics appended successfully!")
    
    # Save Y_true and Y_pred to CSV
    df = pd.DataFrame({
        "y_true": Y_true,
        "y_pred": Y_pred
    })
    df.to_csv(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_test_predictions.csv"))


if __name__ == "__main__":
    main(CONFIG["verbose"])