"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
It loads model checkpoints, applies preprocessing, and saves extracted features for each video clip.
"""

from src.utils.config_utils import load_config
from src.utils.checkpoints_utils import load_checkpoint
from src.Preprocessing.extract_features import extract_features
from src.Preprocessing.volleyball_annot_loader import load_video_annot
from src.utils.logging_utils import setup_logger
from src.baselines.baseline1.finetune_resnet50 import get_model

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

CONFIG = load_config()

logger = setup_logger(
            log_file=__file__,
            log_dir=os.path.join(CONFIG["LOGS_PATH"], "baseline3_logs"),
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
    model = get_model(config=CONFIG, logger=logger, verbose=CONFIG["verbose"])

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
    Main function to extract features from volleyball video clips.

    Iterates through videos and clips, loads annotations, and saves extracted features as .npy files.
    """
    model, transform = prepare_model()

    output_root = CONFIG["DATA_PATHS"]["features_root"]
    videos_root = CONFIG["DATA_PATHS"]["videos_root"]
    annot_root = CONFIG["DATA_PATHS"]["annot_root"]

    videos_dirs = os.listdir(videos_root)
    videos_dirs.sort()

    # Iterate on each video and for each video iterate on each clip
    for idx, video_dir in enumerate(videos_dirs):
        if idx in CONFIG["TARGET_VIDEOS"]["train_ids"] or CONFIG["TARGET_VIDEOS"]["val_ids"]:
            logger.info(f"Working on Video Number: {idx}")
            video_dir_path = os.path.join(videos_root, video_dir)

            if not os.path.isdir(video_dir_path):
                continue

            clips_dir = os.listdir(video_dir_path)
            clips_dir.sort()

            for clip_dir in clips_dir:
                clip_dir_path = os.path.join(video_dir_path, clip_dir)

                if not os.path.isdir(clip_dir_path):
                    continue

                annot_file = os.path.join(annot_root, video_dir, clip_dir, f"{clip_dir}.txt")
                output_file = os.path.join(output_root, video_dir)

                if not os.path.isdir(output_file):
                    os.makedirs(output_file)
                
                output_file = os.path.join(output_file, f"{clip_dir}.npy")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                extract_features(clip_dir_path, annot_file, output_file, model, transform, device, image_level=False, image_classify=True)

if __name__ == "__main__":
    main()