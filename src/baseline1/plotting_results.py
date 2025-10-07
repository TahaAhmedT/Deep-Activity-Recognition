from src.utils.plotting_utils.plotting_utils import plot_results
from src.utils.logging_utils.logging_utils import setup_logger

import pandas as pd

logger = setup_logger(
            log_file=__file__,
            log_dir="logs/baselines_logs/baseline1_logs",
            log_to_console=True,
            use_tqdm=True,
        )


def load_data(metrics_path: str, ys_path: str):
    """Loads metrics and confusion matrix data from CSV files.

    Args:
        metrics_path (str): Path to the metrics CSV file.
        ys_path (str): Path to the CSV file containing y_true and y_pred.

    Returns:
        tuple: Tuple containing metrics DataFrame and confusion matrix DataFrame.
    """
    return pd.read_csv(metrics_path), pd.read_csv(ys_path)


def prepare_results_dict(metrics_data, cm_data):
    """Prepares a dictionary of results for plotting.

    Args:
        metrics_data (pd.DataFrame): DataFrame containing training/testing metrics.
        cm_data (pd.DataFrame): DataFrame containing true and predicted labels.

    Returns:
        dict: Dictionary formatted for the plot_results function.
    """
    results = {"Train_Loss": [metrics_data["train_loss"]],
               "Train_Accuracy": [metrics_data["train_acc"]],
               "Test_Loss": [metrics_data["test_loss"]],
               "Test_Accuracy": [metrics_data["test_acc"]],
               "Test_F1_Score": [metrics_data["test_f1"]],
               "Train_Loss_and_Accuracy": [metrics_data["train_loss"], metrics_data["train_acc"]],
               "Test_Loss_and_Accuracy": [metrics_data["test_loss"], metrics_data["test_acc"]],
               "Train_Test_Loss": [metrics_data["train_loss"], metrics_data["test_loss"]],
               "Train_Test_Accuracy": [metrics_data["train_acc"], metrics_data["test_acc"]],
               "Confusion_Matrix": [cm_data["y_true"], cm_data["y_pred"]]
               }
    return results


def main():
    """Main entry point for plotting baseline1 results.

    Loads data, prepares results, and generates plots for metrics and confusion matrix.
    """
    # Load metrics data and confusion matrix required data (y_true, y_pred)
    logger.info("[INFO] Loading Required Data for Visualization...")
    metrics_data, cm_data = load_data("logs\training_logs\b1_training.csv", "logs\training_logs\b1_test_predictions.csv")
    logger.info("[INFO] Required Data Loaded Successfully!")

    # Prepare results dict to pass to plotting utils
    logger.info("[INFO] Preparing Results' Dictionary...")
    results = prepare_results_dict(metrics_data, cm_data)
    logger.info("[INFO] Results' Dictionary Prepared Successfully!")

    # Pass results to plot_results function
    logger.info("[INFO] Starting Plotting...")
    plot_results(results, "assets\baseline1_plots")
    logger.info("[INFO] Plotting Finished Successfully!")


if __name__ == "__main__":
    main()
