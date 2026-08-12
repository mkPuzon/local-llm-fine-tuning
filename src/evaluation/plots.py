import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(
    metrics,
    ax=None,
):
    """
    Plot the confusion matrix.

    Rows    = actual class
    Columns = predicted class
    """

    if ax is None:
        _, ax = plt.subplots(
            figsize=(7, 6)
        )

    matrix = np.array([
        [
            metrics["true_negative"],
            metrics["false_positive"],
        ],
        [
            metrics["false_negative"],
            metrics["true_positive"],
        ],
    ])

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        square=True,
        cmap="Blues",
        xticklabels=[
            "Accepted (0)",
            "Rejected (1)",
        ],
        yticklabels=[
            "Accepted (0)",
            "Rejected (1)",
        ],
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    return ax