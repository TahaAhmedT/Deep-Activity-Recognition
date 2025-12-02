from ..utils import plot_results, load_data, prepare_results_dict, setup_logger


def visualize(metrics_path, ys_path, save_path, log_dir, verbose):
    """
    Loads data, prepares results, and generates plots for metrics and confusion matrix.
    """
    logger = setup_logger(
    log_file=__file__,
    log_dir=log_dir,
    log_to_console=verbose,
    use_tqdm=True,
    )

    logger.info("Loading data for visualization...")
    metrics_data, cm_data = load_data(metrics_path, ys_path)
    logger.info("Data loaded successfully.")

    logger.info("Preparing results dictionary...")
    results = prepare_results_dict(metrics_data, cm_data)
    logger.info("Results dictionary prepared.")

    logger.info("Generating plots...")
    plot_results(results, save_path)
    logger.info("All plots generated successfully and saved to '%s'.", save_path)
