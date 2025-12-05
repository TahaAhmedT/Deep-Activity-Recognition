"""
This script extracts deep features from volleyball player images using a fine-tuned ResNet-50 model.
"""

from utils import load_config, setup_logger, extract


CONFIG = load_config()

logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline3_logs"],
            log_to_console=CONFIG['verbose'],
            use_tqdm=True
        )


def main():
    """
    Main function to extract features from volleyball video clips using extract_features_utils function.
    """
    # model, transform = prepare_model()

    logger.info("Starting Features Extraction...")
    extract(log_dir=CONFIG["baseline3_logs"],
    videos_root=CONFIG["DATA_PATHS"]["videos_root"],
    train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
    val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
    annot_root=CONFIG["DATA_PATHS"]["annot_root"],
    output_root=CONFIG["DATA_PATHS"]["pooled_players_features_root"],
    num_classes=CONFIG["NUM_LABELS"],
    checkpoint_path="models/b3_models/checkpoints/epoch_2.pth",
    image_level=False,
    image_classify=True,
    verbose=CONFIG["verbose"])

    logger.info("Features Extraction Finished Successfully!")

    
if __name__ == "__main__":
    main()