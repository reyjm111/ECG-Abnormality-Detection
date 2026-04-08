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
    metric_names = ["train_loss", "val_loss", "accuracy", "precision", "recall", "f1", "auc"]
    summary_results = {}
    metric_keys = [k for k in fold_results[0].keys() if k not in ["fold", "best_epoch", "cm"]]

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