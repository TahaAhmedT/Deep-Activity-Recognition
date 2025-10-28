from src.utils.plotting_utils import plot_results
from src.utils.logging_utils import setup_logger
from src.utils.plotting_utils import load_data, prepare_results_dict
import pandas as pd
import os


logger = setup_logger(
    log_file=__file__,
    log_dir="logs/baselines_logs/baseline1_logs",
    log_to_console=True,
    use_tqdm=True,
)


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
