def add_predictions(
    predictions,
    threshold=0.5,
):
    """
    Convert P(class=1) into binary predictions.

    probability_1 >= threshold → 1
    probability_1 < threshold  → 0
    """

    predictions = predictions.copy()

    predictions["predicted_label"] = (
        predictions["probability_1"] >= threshold
    ).astype(int)

    return predictions


def calculate_metrics(predictions):
    """
    Calculate binary classification metrics.

    Class 0 = accepted
    Class 1 = rejected
    """

    y_true = predictions["true_label"]
    y_pred = predictions["predicted_label"]

    # --------------------------------------------------------
    # Confusion matrix components
    # --------------------------------------------------------

    true_positive = (
        (y_true == 1) &
        (y_pred == 1)
    ).sum()

    true_negative = (
        (y_true == 0) &
        (y_pred == 0)
    ).sum()

    false_positive = (
        (y_true == 0) &
        (y_pred == 1)
    ).sum()

    false_negative = (
        (y_true == 1) &
        (y_pred == 0)
    ).sum()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total = len(predictions)

    accuracy = (
        (true_positive + true_negative)
        / total
        if total > 0
        else 0.0
    )

    precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return {
        "n": total,

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,

        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }