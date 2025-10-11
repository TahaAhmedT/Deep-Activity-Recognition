import os
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


def main():
    print("plotting_utils is ready for use.")


if __name__ == "__main__":
    main()
