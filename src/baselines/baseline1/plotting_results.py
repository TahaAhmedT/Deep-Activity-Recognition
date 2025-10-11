from src.utils.plotting_utils import plot_results
from src.utils.logging_utils import setup_logger
import pandas as pd
import os


logger = setup_logger(
    log_file=__file__,
    log_dir="logs/baselines_logs/baseline1_logs",
    log_to_console=True,
    use_tqdm=True,
)


def load_data(metrics_path: str, ys_path: str):
    """
    Loads metrics and confusion matrix data from CSV files.
    """
    metrics_data = pd.read_csv(metrics_path)
    cm_data = pd.read_csv(ys_path)
    return metrics_data, cm_data


def prepare_results_dict(metrics_data, cm_data):
    """
    Prepares a dictionary of results for plotting.
    """
    results = {
        "Train_Loss": [metrics_data["train_loss"].tolist()],
        "Train_Accuracy": [metrics_data["train_acc"].tolist()],
        "Test_Loss": [metrics_data["test_loss"].tolist()],
        "Test_Accuracy": [metrics_data["test_acc"].tolist()],
        "Test_F1_Score": [metrics_data["test_f1"].tolist()],
        "Train_Loss_and_Accuracy": [
            metrics_data["train_loss"].tolist(),
            metrics_data["train_acc"].tolist()
        ],
        "Test_Loss_and_Accuracy": [
            metrics_data["test_loss"].tolist(),
            metrics_data["test_acc"].tolist()
        ],
        "Train_Test_Loss": [
            metrics_data["train_loss"].tolist(),
            metrics_data["test_loss"].tolist()
        ],
        "Train_Test_Accuracy": [
            metrics_data["train_acc"].tolist(),
            metrics_data["test_acc"].tolist()
        ],
        "Confusion_Matrix": (
            cm_data["y_true"].tolist(),
            cm_data["y_pred"].tolist()
        )
    }

    return results


def main():
    """
    Loads data, prepares results, and generates plots for metrics and confusion matrix.
    """
    metrics_path = "logs/training_logs/b1_training.csv"
    ys_path = "logs/training_logs/b1_test_predictions.csv"
    save_path = "assets/baselines_assets/baseline1"

    logger.info("Loading data for visualization...")
    metrics_data, cm_data = load_data(metrics_path, ys_path)
    logger.info("Data loaded successfully.")

    logger.info("Preparing results dictionary...")
    results = prepare_results_dict(metrics_data, cm_data)
    logger.info("Results dictionary prepared.")

    logger.info("Generating plots...")
    plot_results(results, save_path)
    logger.info("All plots generated successfully and saved to '%s'.", save_path)


if __name__ == "__main__":
    main()
