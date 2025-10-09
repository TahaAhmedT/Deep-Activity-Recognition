import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_results(results: dict, save_path: str):
    """Plots various results such as accuracy, loss, and confusion matrix.

    Args:
        results (dict): Dictionary containing result data to plot.
        save_path (str): Directory path to save the plots.
    """
    # Plot Line Plots
    for name, vals in results.items():
        if name in ["Train_Accuracy", "Train_Loss", "Test_Accuracy", "Test_Loss", "Test_F1_score"]:
            line_plot(results[name], [name], save_path, name, name, "epoch")
        elif name == "Train_Loss_and_Accuracy":
            line_plot(results[name], ["train_loss", "train_accuracy"], save_path, name, name, "epoch")
        elif name == "Test_Loss_and_Accuracy":
            line_plot(results[name], ["test_loss", "test_accuracy"], save_path, name, name, "epoch")
        elif name == "Train_Test_Loss":
            line_plot(results[name], ["train_loss", "test_loss"], save_path, name, name, "epoch")
        elif name == "Train_Test_Accuracy":
            line_plot(results[name], ["train_accuracy", "test_accuracy"], save_path, name, name, "epoch")
        elif name == "Confusion_Matrix":
            confusion_matrix(results[name], save_path, name)


def line_plot(data: list, label: list[str], save_path: str, figname: str, title: None, xlabel: None, ylabel: None):
    """Plots line graphs for the provided data and saves the figure.

    Args:
        data (list): List of data series to plot.
        label (list[str]): List of labels for each data series.
        save_path (str): Directory path to save the plot.
        figname (str): Name for the saved plot file.
        title (None): Title of the plot.
        xlabel (None): Label for the x-axis.
        ylabel (None): Label for the y-axis.
    """
    for i in range(len(label)):
        plt.plot(data[i], label=label[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(os.path.join(save_path, f"{figname}_plot.png"))


def confusion_matrix(data, save_path: str, figname: str):
    """Plots and saves a confusion matrix.

    Args:
        data: Tuple or list containing true and predicted labels.
        save_path (str): Directory path to save the confusion matrix plot.
        figname (str): Name for the saved confusion matrix file.
    """
    cm = confusion_matrix(data[0], data[1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot().figure_.savefig(os.path.join(save_path, f"{figname}.png"))


def main():
    print("Welcome From Main Function...")

if __name__ == "__main__":
    main()