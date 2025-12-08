"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
"""

from utils.config_utils import load_config
from utils.logging_utils import setup_logger
from utils.extract_features_utils import extract

CONFIG = load_config()

logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline4_logs"],
            log_to_console=CONFIG['verbose'],
            use_tqdm=True
        )

def main():
    """
    Main function to extract features from volleyball video clips using extract_features_helper function.
    """

    logger.info("Starting Features Extraction...")
    extract(log_dir=CONFIG["baseline4_logs"],
    videos_root=CONFIG["DATA_PATHS"]["videos_root"],
    train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
    val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
    annot_root=CONFIG["DATA_PATHS"]["annot_root"],
    output_root=CONFIG["DATA_PATHS"]["frame_features_root"],
    num_classes=CONFIG["NUM_LABELS"],
    checkpoint_path="models/b1_models/checkpoints/epoch_5.pth",
    image_level=True,
    image_classify=False,
    verbose=CONFIG["verbose"])

    logger.info("Features Extraction Finished Successfully!")

    
if __name__ == "__main__":
    main()