"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
"""

from ...utils import load_config, load_checkpoint, setup_logger
from ...helpers import get_resnet_model, extract

import torch
import torch.nn as nn
import torchvision.transforms as transforms


CONFIG = load_config()

logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline4_logs"],
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
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Check if a GPU is available if not, use a CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device:{device}")

    # Load ResNet-50 model with pretrained weights
    model = get_resnet_model(logger=logger, num_classes=CONFIG["NUM_LABELS"], verbose=CONFIG["verbose"])

    # Load a checkpoint saved during training
    logger.info("Loading the Model's Checkpoint...")
    checkpoint_path = "models/b1_models/checkpoints/epoch_5.pth"
    checkpoint = torch.load(checkpoint_path)

    # Load trained weights into the model
    model = load_checkpoint(checkpoint=checkpoint, model=model)
    
    # Remove the classification head (i.e., the fully connected layers)
    model = nn.Sequential(*(list(model.children())[:-1]))

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
    extract(log_dir=CONFIG["baseline4_logs"],
    videos_root=CONFIG["DATA_PATHS"]["videos_root"],
    train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
    val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
    annot_root=CONFIG["DATA_PATHS"]["annot_root"],
    output_root=CONFIG["DATA_PATHS"]["frame_features_root"],
    model=model,
    transform=transform,
    image_level=True,
    image_classify=False,
    verbose=CONFIG["verbose"])

    logger.info("Features Extraction Finished Successfully!")

    
if __name__ == "__main__":
    main()