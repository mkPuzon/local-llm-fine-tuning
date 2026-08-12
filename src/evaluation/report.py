import json
from pathlib import Path

import matplotlib.pyplot as plt

from .plots import plot_confusion_matrix


def print_report(metrics):
    """
    Print a human-readable evaluation report.
    """

    def percentage(value):
        return f"{value * 100:.2f}%"

    print()
    print("=" * 60)
    print("              BINARY CLASSIFIER EVALUATION")
    print("=" * 60)

    print()
    print(
        f"Examples evaluated: "
        f"{metrics['n']:,}"
    )

    print()
    print("METRICS")
    print("-" * 60)

    print(
        f"Accuracy:            "
        f"{percentage(metrics['accuracy'])}"
    )

    print(
        f"Precision:           "
        f"{percentage(metrics['precision'])}"
    )

    print(
        f"Recall:              "
        f"{percentage(metrics['recall'])}"
    )

    print(
        f"False Positive Rate: "
        f"{percentage(metrics['false_positive_rate'])}"
    )

    print()
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(
        f"True negatives:  "
        f"{metrics['true_negative']}"
    )

    print(
        f"False positives: "
        f"{metrics['false_positive']}"
    )

    print(
        f"False negatives: "
        f"{metrics['false_negative']}"
    )

    print(
        f"True positives:  "
        f"{metrics['true_positive']}"
    )

    print()
    print("=" * 60)
    print()


def save_metrics(
    metrics,
    output_path,
):
    """
    Save metrics to a JSON file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )


def save_confusion_matrix(
    metrics,
    output_path,
):
    """
    Generate and save the confusion matrix plot.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    plot_confusion_matrix(
        metrics,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)