import numpy as np
import pandas as pd


def accuracy_score(
    predictions,
    labels
):

    labels = labels.astype(str).to_numpy()

    return float(
        (predictions == labels).mean()
    )


def confusion_matrix(
    predictions,
    labels,
    class_labels
):

    matrix = pd.DataFrame(
        0,
        index=class_labels,
        columns=class_labels,
        dtype=int
    )

    for predicted, actual in zip(
        predictions,
        labels.astype(str)
    ):
        matrix.loc[actual, predicted] += 1

    return matrix


def classification_report(
    predictions,
    labels,
    class_labels
):

    matrix = confusion_matrix(
        predictions,
        labels,
        class_labels
    )

    per_class = {}

    for label in class_labels:

        true_positive = int(
            matrix.loc[label, label]
        )

        false_positive = int(
            matrix[label].sum()
            -
            true_positive
        )

        false_negative = int(
            matrix.loc[label].sum()
            -
            true_positive
        )

        precision = (
            true_positive
            /
            (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0
        )

        recall = (
            true_positive
            /
            (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0
        )

        f1_score = (
            2 * precision * recall
            /
            (precision + recall)
            if precision + recall > 0
            else 0
        )

        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "support": int(matrix.loc[label].sum()),
        }

    accuracy = accuracy_score(
        predictions,
        labels
    )

    macro_f1 = float(
        np.mean(
            [
                values["f1_score"]
                for values in per_class.values()
            ]
        )
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": matrix.to_numpy().tolist(),
        },
    }
