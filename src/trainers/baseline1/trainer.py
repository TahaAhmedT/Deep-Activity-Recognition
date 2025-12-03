"""
Entry point script to fine-tune a ResNet-50 model at frame (image) level for baseline1.

This module:
- Loads project configuration and sets up logging.
- Calls the finetune helper to train a ResNet-50 model on image-level data (central frames per clip).
- After training, calls the visualize helper to produce plots for training metrics and confusion matrix.

The script is intended to be executed as a standalone program.
"""
from ...utils import load_config, setup_logger
from ...helpers import finetune, visualize


def main():
    """Main entry point to fine-tune ResNet-50 for image-level classification.

    This function performs the following steps:
        1. Loads configuration via load_config().
        2. Sets up a logger using setup_logger().
        3. Calls finetune(...) with parameters read from the configuration to train the model.
        4. Calls visualize(...) to generate plots and save visualization artifacts.

    The function does not return a value; all outputs (models, logs, plots) are persisted to disk
    as specified in the configuration.
    """
    CONFIG = load_config()
    logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline1_logs"],
            log_to_console=CONFIG["verbose"],
            use_tqdm=True,
        )
    
    logger.info("Starting Fine-tuning ResNet50 on Image Dataset (Frame Level)...")
    finetune(
        log_dir=CONFIG["baseline1_logs"],
        lr=CONFIG["TRAINING_PARAMS"]["lr"],
        num_epochs=CONFIG["TRAINING_PARAMS"]["num_epochs"],
        batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"],
        train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
        val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
        features=False,
        model_name="resnet",
        num_classes=CONFIG["NUM_LABELS"],
        actions_dict=CONFIG["CATEGORIES_DICT"],
        metrics_logs="logs/training_logs/b1_training.csv",
        preds_logs="logs/training_logs/b1_test_predictions.csv",
        save_path="models/b1_models/checkpoints",
        image_level=True,
        verbose=CONFIG["verbose"]
        )

    logger.info("Fine-tuning ResNet50 Finished Successfully!")

    logger.info("Visualizing Results...")
    visualize(
        metrics_path="logs/training_logs/b1_training.csv",
        ys_path="logs/training_logs/b1_test_predictions.csv",
        save_path="assets/baselines_assets/baseline1",
        log_dir=CONFIG["baseline1_logs"],
        verbose=CONFIG["verbose"]
        )

    logger.info("Visualization Finished Successfully!")

if __name__ == "__main__":
    main()
