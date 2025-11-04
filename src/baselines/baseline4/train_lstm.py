from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logger
from src.helpers.finetune_helper import finetune
from src.helpers.visualize_helper import visualize


def main():
    CONFIG = load_config()
    logger = setup_logger(
            log_file=__file__,
            log_dir=CONFIG["baseline4_logs"],
            log_to_console=CONFIG["verbose"],
            use_tqdm=True,
        )
    
    logger.info("Starting Training Group Activity Temporal Classifier on Features' Dataset (Frame Level)...")
    finetune(log_dir=CONFIG["baseline4_logs"],
            lr=CONFIG["TRAINING_PARAMS"]["lr"],
            num_epochs=CONFIG["TRAINING_PARAMS"]["num_epochs"],
            batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
            videos_root=CONFIG["DATA_PATHS"]["videos_root"],
            annot_root=CONFIG["DATA_PATHS"]["annot_root"],
            train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
            val_ids=CONFIG["TARGET_VIDEOS"]["val_ids"],
            features=True,
            model_name="lstm",
            num_classes=CONFIG["NUM_LABELS"],
            actions_dict=CONFIG["CATEGORIES_DICT"],
            metrics_logs="logs/training_logs/b4_training.csv",
            preds_logs="logs/training_logs/b4_test_predictions.csv",
            save_path="models/b4_models/checkpoints",
            output_file=CONFIG["DATA_PATHS"]["frame_features_root"],
            input_size=CONFIG["EXTRACTED_FEATURES_SIZE"],
            hidden_size=CONFIG["HIDDEN_SIZE"],
            num_layers=CONFIG["NUM_LAYERS"],
            verbose=CONFIG["verbose"]
        )

    logger.info("Training Group Activity Temporal Classifier Finished Successfully!")

    logger.info("Visualizing Results...")
    visualize(metrics_path="logs/training_logs/b4_training.csv",
            ys_path="logs/training_logs/b4_test_predictions.csv",
            save_path="assets/baselines_assets/baseline4",
            log_dir=CONFIG["baseline4_logs"],
            verbose=CONFIG["verbose"]
        )

    logger.info("Visualization Finished Successfully!")
    

