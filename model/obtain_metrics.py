import matplotlib.pyplot as plt
import numpy as np

def obtain_metrics(fold_results, history_results):

    """
    Function:
        Extracting metrics from CNN model.

    Parameters: 
        fold_results (list): List of best epoch results in each fold.
        history_results (list): List of fold's epoch performance. 

    Outputs:
        Printed fold summary results and plotting training and validation loss and other metrics.
    """

    metric_names = ["train_loss", "val_loss", "accuracy", "precision", "recall", "F1", "AUC"]
    summary_results = {}
    metric_keys = [k for k in fold_results[0].keys() if k not in ["fold", "best_epoch", "cm"]]

    # Obtain mean and stdev out of each metric across folds
    for key in metric_keys:
        values = [fold[key] for fold in fold_results if not np.isnan(fold[key])]
        summary_results[key] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }

    print("Fold summary:")
    print(summary_results)

    n_epochs = len(history_results[0]["history"])
    epochs = np.arange(1, n_epochs + 1)

    # Loop through each metric and obtain the average across folds per epoch
    for metric in metric_names:
        metric_matrix = []

        for fold in history_results:
            values = [epoch_result[metric] for epoch_result in fold["history"]]
            metric_matrix.append(values)

        metric_matrix = np.array(metric_matrix, dtype=float)
        mean_values = np.nanmean(metric_matrix, axis=0)

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, mean_values)
        plt.title(f"Epoch vs {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.show()

    cms = np.array([fold["cm"] for fold in fold_results], dtype=float)

    # Sum confusion matrix across folds
    cm_sum = np.sum(cms, axis=0)

    # Mean confusion matrix across folds
    cm_mean = np.mean(cms, axis=0)

    def plot_confusion_matrix(cm, title="Confusion Matrix", fmt=".0f"):
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation="nearest")
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks([0, 1], ["Normal", "Abnormal"])
        plt.yticks([0, 1], ["Normal", "Abnormal"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         ha="center", va="center")

        plt.colorbar()
        plt.tight_layout()
        plt.show()

    plot_confusion_matrix(cm_sum, title="Summed Confusion Matrix Across Folds", fmt=".0f")
    plot_confusion_matrix(cm_mean, title="Mean Confusion Matrix Across Folds", fmt=".2f")