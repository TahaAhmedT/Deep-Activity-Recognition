from utils.config_utils import load_config
from utils.test_utils import test

def main():
    """
    Main entry point to test B1 Model.
    """

    CONFIG = load_config()
    
    test(log_dir=CONFIG["baseline1_logs"],
        batch_size=CONFIG["TRAINING_PARAMS"]["batch_size"],
        videos_root=CONFIG["DATA_PATHS"]["videos_root"],
        annot_root=CONFIG["DATA_PATHS"]["annot_root"],
        train_ids=CONFIG["TARGET_VIDEOS"]["train_ids"],
        test_ids=CONFIG["TARGET_VIDEOS"]["test_ids"],
        features=False,
        model_name="b1",
        checkpoint="models/b1/best_model.pth",
        num_classes=CONFIG["NUM_LABELS"],
        actions_dict=CONFIG["CATEGORIES_DICT"],
        save_path="assets/test_assets",
        figname="B1_Confusion_Matrix",
        image_level=True,
        crop=False,
        verbose=CONFIG["verbose"]
    )


if __name__ == "__main__":
    main()