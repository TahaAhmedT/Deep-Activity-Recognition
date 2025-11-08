from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logger
from src.helpers.extract_features_helper import extract


def main():
    CONFIG = load_config()
    logger = setup_logger(
          log_file=__file__,
          log_dir=CONFIG["baseline5_logs"],
          log_to_console=CONFIG["verbose"],
          use_tqdm=True
    )
    logger.info("Starting Features Extraction (player-level)...")
    extract(
        log_dir=CONFIG["baseline5_logs"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
        val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"],
        output_root=CONFIG["DATA_PATHS"]["player_features_root"],
        num_classes=CONFIG["NUM_ACTIONS"],
        checkpoint_path="",
        image_level=False,
        image_classify=False,
        verbose=CONFIG["verbose"]
        )
    logger.info("Features Extraction Finished Successfully!")


if __name__ == "__main__":
    main()