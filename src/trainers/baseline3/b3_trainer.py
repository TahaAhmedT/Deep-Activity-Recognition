"""
Entry point script to train an ANN classifier on pooled player features for baseline3.

This module:
- Loads project configuration and sets up logging.
- Calls the finetune helper to train an ANN model on pre-extracted player-level features
  (pooled features for 12 players).
- Calls the visualize helper to generate plots for training metrics and confusion matrix.

The script is intended to be executed as a standalone program.
"""
from utils.logging_utils import setup_logger
from utils.config_utils import load_config
from utils.finetune_utils import finetune
from utils.visualize_utils import visualize

import os

def main():
    """Main entry point to train the ANN classifier on extracted player features.

    Steps performed:
        1. Load configuration via load_config().
        2. Initialize a logger using setup_logger().
        3. Call finetune(...) with features=True and ANN-specific parameters to train the model.
        4. Invoke visualize(...) to save training plots and prediction artifacts.

    The function persists models, logs and visualizations to disk as configured and does not return a value.
    """
    CONFIG = load_config()
    logger = setup_logger(
            log_file=__file__,
            log_dir=os.path.join(CONFIG["baseline3_logs"], "exp2"),
            log_to_console=CONFIG["verbose"],
            use_tqdm=True,
        )
    
    logger.info("Starting Training Group Activity Classifier on Image's features Dataset (12 players pooled features)...")
    finetune(log_dir=os.path.join(CONFIG["baseline3_logs"], "exp2"),
            lr=CONFIG["TRAINING_PARAMS"]["lr"],
            num_epochs=CONFIG["TRAINING_PARAMS"]["num_epochs"],
            batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
            videos_root=CONFIG["DATA_PATHS"]["videos_root"],
            annot_root=CONFIG["DATA_PATHS"]["annot_root"],
            train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
            val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
            features=True,
            model_name="b3",
            num_classes=CONFIG["NUM_LABELS"],
            actions_dict=CONFIG["CATEGORIES_DICT"],
            metrics_logs="logs/training_logs/baseline3/exp2/b3_ann_training.csv",
            preds_logs="logs/training_logs/baseline3/exp2/b3_ann_test_predictions.csv",
            save_path="models/b3",
            use_scheduler=CONFIG["TRAINING_PARAMS"]["use_scheduler"],
            image_level=False,
            crop=False,
            output_file=CONFIG["DATA_PATHS"]["pooled_players_features_root"],
            input_size=CONFIG["EXTRACTED_FEATURES_SIZE"],
            verbose=CONFIG["verbose"]
        )

    logger.info("Training Group Activity Classifier Finished Successfully!")

    logger.info("Visualizing Results...")
    visualize(metrics_path="logs/training_logs/baseline3/exp2/b3_ann_training.csv",
            ys_path="logs/training_logs/baseline3/exp2/b3_ann_test_predictions.csv",
            save_path="assets/baselines_assets/baseline3",
            log_dir=os.path.join(CONFIG["baseline3_logs"], "exp2"),
            verbose=CONFIG["verbose"]
        )

    logger.info("Visualization Finished Successfully!")


if __name__ == "__main__":
    main()
