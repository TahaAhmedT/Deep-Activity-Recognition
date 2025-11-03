from src.utils.logging_utils import setup_logger
from src.utils.config_utils import load_config
from src.baselines.baseline3.ann_model import ANN
from src.helpers.datasets import FeaturesDataset
from src.utils.train_utils import train_step
from src.utils.test_utils import test_step
from src.utils.checkpoints_utils import save_checkpoint
from src.helpers.finetune_helper import set_all_seeds, get_optimizer

import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import csv

CONFIG = load_config()

logger = setup_logger(
    log_file=__file__,
    log_dir=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
    log_to_console=CONFIG["verbose"],
    use_tqdm=True
)

def get_data_loader():
    logger.info("Creating Data Loaders...")
    batch_size = CONFIG["TRAINING_PARAMS"]["batch_size"]
    train_dataset = FeaturesDataset(
        output_file=CONFIG["DATA_PATHS"]["features_root"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        target_videos=CONFIG["TARGET_VIDEOS"]["train_ids"]
    )
    test_dataset = FeaturesDataset(
        output_file=CONFIG["DATA_PATHS"]["features_root"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        target_videos=CONFIG["TARGET_VIDEOS"]["val_ids"]
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info("Data Loaders Created.")
    return trainloader, testloader

def get_model():
    logger.info("Initializing ANN Model...")
    model = ANN(input_size=CONFIG["EXTRACTED_FEATURES_SIZE"], n_classes=CONFIG["NUM_LABELS"])
    logger.info("ANN Model Initialized.")
    return model

def main():
    set_all_seeds(42, logger=logger)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")
    trainloader, testloader = get_data_loader()
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer, _ = get_optimizer(model=model, config=CONFIG, logger=logger)
    
    num_epochs = CONFIG["TRAINING_PARAMS"]["num_epochs"]
    logger.info(f"Starting Training and Testing with Number of Epochs = {num_epochs}.")

    # Create or open a CSV file and define headers
    logger.info("Opening a CSV file to log training metrics.")
    with open(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_ann_training.csv"), mode="w", newline="") as f:
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
            device=device,
            logs_path=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
            verbose=CONFIG["verbose"]
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
            save_checkpoint(checkpoint, os.path.join(CONFIG["MODELS_PATH"], f"b3_ann_models/checkpoints/epoch_{epoch+1}.pth"))
            logger.info(f"Model checkpoint saved at epoch {epoch+1}/{num_epochs}.")

        
        # Testing step
        logger.info("Starting testing step...")
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
        # scheduler.step(test_loss)
        logger.info("Testing step completed.")

        # Append results to CSV
        logger.info("Appending the epoch's metrics to CSV.")
        with open(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_ann_training.csv"), mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc, test_f1_score.item()])
        logger.info("Epoch's metrics appended successfully!")

    # Save Y_true and Y_pred to CSV (to visualize confusion matrix later)
    df = pd.DataFrame({
        "y_true": Y_true,
        "y_pred": Y_pred
    })
    df.to_csv(os.path.join(CONFIG["TRAINING_LOGS_PATH"], "b3_ann_test_predictions.csv"))

if __name__ == "__main__":
    main()
    