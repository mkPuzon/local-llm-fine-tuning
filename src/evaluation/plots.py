import json
import matplotlib.pyplot as plt

def plot_loss(trainer, show_val=True):
    history = trainer.state.log_history

    train_steps = []
    train_loss = []
    val_steps = []
    val_loss = []

    for entry in history:
        if "loss" in entry and "step" in entry:
            train_steps.append(entry["step"])
            train_loss.append(entry["loss"])

        if show_val and "eval_loss" in entry and "step" in entry:
            val_steps.append(entry["step"])
            val_loss.append(entry["eval_loss"])

    plt.figure(figsize=(10, 6))

    plt.plot(train_steps, train_loss, label="Training Loss")

    if show_val and val_loss:
        plt.plot(val_steps, val_loss, label="Validation Loss")

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss" + (" vs. Validation Loss" if val_loss else ""))
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_model_metrics(model_names, results_dir="../results"):
    """
    Plot accuracy, precision, TPR, and FPR for a list of model result files.

    Parameters
    ----------
    model_names : list[str]
        Model names corresponding to JSON files in `results_dir`.

    results_dir : str
        Directory containing the result JSON files.
    """

    results = []

    for model_name in model_names:
        path = f"{results_dir}/{model_name}.json"

        with open(path, "r") as f:
            results.append(json.load(f))

    labels = [r["model"] for r in results]

    metrics = {
        "Accuracy": [r["accuracy"] for r in results],
        "Precision": [r["precision"] for r in results],
        "TPR / Recall": [r["tpr"] for r in results],
        "FPR": [r["fpr"] for r in results],
    }

    colors = [
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12),
    )

    for ax, (title, values), color in zip(
        axes.flat,
        metrics.items(),
        colors,
    ):
        bars = ax.bar(
            labels,
            values,
            color=color,
        )

        ax.set_title(title, fontsize=14, pad=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.1)

        ax.tick_params(axis="x", rotation=45)

        # Add value labels above each bar
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

        ax.grid(
            axis="y",
            alpha=0.3,
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.12,
        top=0.93,
        wspace=0.30,
        hspace=0.35,
    )

    fig.suptitle(
        "Classification Model Evaluation Metrics",
        fontsize=18,
        y=0.97,
    )

    plt.show()