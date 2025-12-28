from utils.config_utils import load_config
from utils.test_utils import test

def main():
    """
    Main entry point to test B8 Model.
    """

    CONFIG = load_config()
    
    test(log_dir=CONFIG["baseline8_logs"],
        batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"],
        train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
        test_ids=CONFIG["TARGET_VIDEOS"]["test_ids"],
        features=True,
        model_name="b8",
        checkpoint="models/b8/best_model.pth",
        num_classes=CONFIG["NUM_LABELS"],
        actions_dict=CONFIG["CATEGORIES_DICT"],
        save_path="assets/test_assets",
        figname="B8_Confusion_Matrix",
        crop=True,
        output_file=CONFIG["DATA_PATHS"]["player_features_root"],
        input_size=CONFIG["EXTRACTED_FEATURES_SIZE"],
        hidden_size1=CONFIG["HIDDEN_SIZE1"],
        hidden_size2=CONFIG["HIDDEN_SIZE2"],
        num_layers=CONFIG["NUM_LAYERS"],
        sequence=True,
        verbose=CONFIG["verbose"]
    )


if __name__ == "__main__":
    main()