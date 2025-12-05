from .logging_utils import setup_logger

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, ConfusionMatrixDisplay


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_results(results: dict, save_path: str):
    """
    Plots various results such as accuracy, loss, and confusion matrix.

    Args:
        results (dict): Dictionary containing result data to plot.
        save_path (str): Directory path to save the plots.
    """
    ensure_dir(save_path)

    for name, vals in results.items():
        # Line plots for individual metrics
        if name in ["Train_Accuracy", "Train_Loss", "Test_Accuracy", "Test_Loss", "Test_F1_Score"]:
            line_plot(
                data=vals,
                labels=[name],
                save_path=save_path,
                figname=name,
                title=name.replace("_", " "),
                xlabel="Epoch",
                ylabel=name.split("_")[-1]
            )

        # Combined plots
        elif name == "Train_Loss_and_Accuracy":
            line_plot(
                data=vals,
                labels=["Train Loss", "Train Accuracy"],
                save_path=save_path,
                figname=name,
                title="Training Loss & Accuracy",
                xlabel="Epoch"
            )

        elif name == "Test_Loss_and_Accuracy":
            line_plot(
                data=vals,
                labels=["Test Loss", "Test Accuracy"],
                save_path=save_path,
                figname=name,
                title="Testing Loss & Accuracy",
                xlabel="Epoch"
            )

        elif name == "Train_Test_Loss":
            line_plot(
                data=vals,
                labels=["Train Loss", "Test Loss"],
                save_path=save_path,
                figname=name,
                title="Train vs Test Loss",
                xlabel="Epoch"
            )

        elif name == "Train_Test_Accuracy":
            line_plot(
                data=vals,
                labels=["Train Accuracy", "Test Accuracy"],
                save_path=save_path,
                figname=name,
                title="Train vs Test Accuracy",
                xlabel="Epoch"
            )

        elif name == "Confusion_Matrix":
            y_true, y_pred = vals
            plot_confusion_matrix(y_true, y_pred, save_path, figname=name)


def line_plot(data: list, labels: list[str], save_path: str, figname: str,
              title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Plots line graphs for the provided data and saves the figure.
    """
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(data[0]) + 1)

    for i, label in enumerate(labels):
        plt.plot(epochs, data[i], label=label, linewidth=2)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f"{figname}.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path: str, figname: str):
    """
    Plots and saves a confusion matrix.
    """
    cm = sk_confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{figname}.png"))
    plt.close()


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



def main():
    print("plotting_utils is ready for use.")


if __name__ == "__main__":
    main()
