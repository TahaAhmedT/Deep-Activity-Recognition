"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
It loads model checkpoints, applies preprocessing, and saves extracted features for each video clip.
"""

from src.utils.config_utils import load_config
from src.utils.checkpoints_utils import load_checkpoint
from src.Preprocessing.extract_features import extract_features
from src.Preprocessing.volleyball_annot_loader import load_video_annot
from src.utils.logging_utils import setup_logger
from src.helpers.finetune_helper import get_model
from src.helpers.extract_features_helper import extract

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

CONFIG = load_config()

logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline3_logs"],
            log_to_console=CONFIG['verbose'],
            use_tqdm=True
        )

def prepare_model():
    """
    Prepares the ResNet-50 model for feature extraction.

    Loads the model, applies preprocessing transforms, loads trained weights, and sets the model to evaluation mode.

    Returns:
        tuple: (model, transform) where model is the feature extractor and transform is the preprocessing pipeline.
    """
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    # Check if a GPU is available if not, use a CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device:{device}")

    # Load ResNet-50 model with pretrained weights
    model = get_model(logger=logger, num_classes=CONFIG["NUM_CLASSES"], verbose=CONFIG["verbose"])

    # Load a checkpoint saved during training
    logger.info("Loading the Model's Checkpoint...")
    checkpoint_path = os.path.join(CONFIG["MODELS_PATH"], "b3_models/checkpoints/epoch_2.pth")
    checkpoint = torch.load(checkpoint_path)

    # Load trained weights into the model
    model = load_checkpoint(checkpoint=checkpoint, model=model)
    
    # Send model to the device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    logger.info(f"The Model is Ready and Sent to {device} Device, and Set to Eval Mode.")

    return model, transform

def main():
    """
    Main function to extract features from volleyball video clips using extract_features_helper function.
    """
    model, transform = prepare_model()

    logger.info("Starting Features Extraction...")
    extract(log_dir=CONFIG["baseline3_logs"],
    videos_root=CONFIG["DATA_PATHS"]["videos_root"],
    train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
    val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
    annot_root=CONFIG["DATA_PATHS"]["annot_root"],
    output_root=CONFIG["DATA_PATHS"]["features_root"],
    model=model,
    transform=transform,
    image_level=False,
    image_classify=True,
    verbose=CONFIG["verbose"])

    logger.info("Features Extraction Finished Successfully!")

    
if __name__ == "__main__":
    main()