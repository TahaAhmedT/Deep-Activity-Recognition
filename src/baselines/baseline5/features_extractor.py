"""
Script to extract per-player features from volleyball clips using a fine-tuned ResNet-50.

This module:
- Loads project configuration and sets up logging
- Calls the extract helper to:
  1. Load a trained ResNet-50 model
  2. Process each clip frame-by-frame
  3. Extract features for each player using tracking annotations
  4. Save the extracted features as .npy files

The extracted features will be used to train the temporal classifier in baseline5.
"""
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logger
from src.helpers.extract_features_helper import extract

import os


def main():
    """Main entry point for player-level feature extraction.

    This function:
        1. Loads configuration via load_config()
        2. Sets up logging with setup_logger()
        3. Calls extract() to process clips and save player features
        
    The extracted features are saved to the directory specified in the config
    under player_features_root. Features are organized by video and clip ID.

    The function does not return a value; outputs are saved directly to disk.
    """
    CONFIG = load_config()
    logger = setup_logger(
          log_file=__file__,
          log_dir=os.path.join(CONFIG["baseline5_logs"], "exp1"),
          log_to_console=CONFIG["verbose"],
          use_tqdm=True
    )
    logger.info("Starting Features Extraction (player-level)...")
    extract(
        log_dir=os.path.join(CONFIG["baseline5_logs"], "exp1"),
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
        val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"],
        output_root=CONFIG["DATA_PATHS"]["player_features_root"],
        num_classes=CONFIG["NUM_ACTIONS"],
        checkpoint_path="models/b3_models/checkpoints/epoch_2.pth",
        image_level=False,
        image_classify=False,
        verbose=CONFIG["verbose"]
        )
    logger.info("Features Extraction Finished Successfully!")


if __name__ == "__main__":
    main()