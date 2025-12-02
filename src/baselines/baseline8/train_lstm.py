"""
Entry point script to train the two-stage pooled-teams LSTM temporal classifier for baseline8.

This module:
- Loads project configuration and sets up logging.
- Calls the finetune helper to train a two-stage sequence model that first models
  per-player temporal dynamics, pool players by team, and then frame-level temporal dynamics.
- Calls the visualize helper to generate plots for training metrics and confusion matrix.

The script is intended to be executed as a standalone program.
"""
from ...utils import load_config, setup_logger
from ...helpers import finetune, visualize

import os

def main():
    """Main entry point to train the Two-Stage Group Activity Temporal Classifier.

    This function performs the following steps:
        1. Load configuration via load_config().
        2. Initialize a logger using setup_logger().
        3. Call finetune(...) with sequence=True and two-stage LSTM-specific parameters
           to train the model on player-level features (features=True).
        4. Call visualize(...) to save training plots and prediction artifacts.

    The function persists models, logs and visualizations to disk as configured and does not return a value.
    """
    CONFIG = load_config()
    logger = setup_logger(
            log_file=__file__,
            log_dir=os.path.join(CONFIG["baseline8_logs"], "exp1"),
            log_to_console=CONFIG["verbose"],
            use_tqdm=True,
        )
    
    logger.info("Starting Training Two Stage Group Activity Temporal Classifier on Features' Dataset (Player-level, pool each team, then frame-level)...")
    finetune(log_dir=os.path.join(CONFIG["baseline8_logs"], "exp1"),
            lr=CONFIG["TRAINING_PARAMS"]["lr"],
            num_epochs=CONFIG["TRAINING_PARAMS"]["num_epochs"],
            batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
            videos_root=CONFIG["DATA_PATHS"]["videos_root"],
            annot_root=CONFIG["DATA_PATHS"]["annot_root"],
            train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
            val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
            features=True,
            model_name="lstm4",
            num_classes=CONFIG["NUM_LABELS"],
            actions_dict=CONFIG["CATEGORIES_DICT"],
            metrics_logs="logs/training_logs/baseline8/exp1/b8_training.csv",
            preds_logs="logs/training_logs/baseline8/exp1/b8_test_predictions.csv",
            save_path="models/b8_models/exp1",
            crop=True,
            output_file=CONFIG["DATA_PATHS"]["player_features_root"],
            input_size=CONFIG["EXTRACTED_FEATURES_SIZE"],
            hidden_size1=CONFIG["HIDDEN_SIZE1"],
            hidden_size2=CONFIG["HIDDEN_SIZE2"],
            num_layers=CONFIG["NUM_LAYERS"],
            sequence=True,
            verbose=CONFIG["verbose"]
        )

    logger.info("Training Two Stage pooled-teams Group Activity Temporal Classifier Finished Successfully!")

    logger.info("Visualizing Results...")
    visualize(metrics_path="logs/training_logs/baseline8/exp1/b8_training.csv",
            ys_path="logs/training_logs/baseline8/exp1/b8_test_predictions.csv",
            save_path="assets/baselines_assets/baseline8/exp1",
            log_dir=os.path.join(CONFIG["baseline8_logs"], "exp1"),
            verbose=CONFIG["verbose"]
        )

    logger.info("Visualization Finished Successfully!")


if __name__ == "__main__":
    main()
