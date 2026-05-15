import math

# pyrefly: ignore [missing-import]
import numpy as np

from src.config import CATEGORICAL_COLUMNS
from src.config import NUMERIC_COLUMNS


class NaiveBayes:

    def __init__(
        self,
        alpha=1,
        variance_smoothing=1e-9
    ):
        self.alpha = alpha
        self.variance_smoothing = variance_smoothing

    def fit(
        self,
        features,
        labels
    ):

        self.classes = sorted(
            labels.astype(str).unique().tolist()
        )

        self.class_log_prior = {}
        self.numeric_stats = {}
        self.category_log_probability = {}
        self.unknown_category_log_probability = {}
        self.categories = {}

        total = len(labels)

        for label in self.classes:

            class_rows = features[
                labels == label
            ]

            class_count = len(class_rows)

            self.class_log_prior[label] = math.log(
                class_count / total
            )

            self.numeric_stats[label] = {}

            for column in NUMERIC_COLUMNS:

                values = class_rows[column].astype(float)

                self.numeric_stats[label][column] = {
                    "mean": float(values.mean()),
                    "variance": float(
                        values.var(ddof=0)
                        +
                        self.variance_smoothing
                    ),
                }

        for column in CATEGORICAL_COLUMNS:
            self.categories[column] = sorted(
                features[column]
                .astype(str)
                .unique()
                .tolist()
            )

        for label in self.classes:

            class_rows = features[
                labels == label
            ]

            self.category_log_probability[label] = {}
            self.unknown_category_log_probability[label] = {}

            for column in CATEGORICAL_COLUMNS:

                categories = self.categories[column]

                counts = (
                    class_rows[column]
                    .astype(str)
                    .value_counts()
                )

                denominator = (
                    len(class_rows)
                    +
                    self.alpha * (len(categories) + 1)
                )

                self.category_log_probability[label][column] = {
                    category: math.log(
                        (
                            counts.get(category, 0)
                            +
                            self.alpha
                        )
                        /
                        denominator
                    )
                    for category in categories
                }

                self.unknown_category_log_probability[label][column] = (
                    math.log(
                        self.alpha / denominator
                    )
                )

    def predict_one(
        self,
        row
    ):

        scores = {}

        for label in self.classes:

            score = self.class_log_prior[label]

            for column in NUMERIC_COLUMNS:

                value = float(row[column])
                mean = self.numeric_stats[label][column]["mean"]
                variance = self.numeric_stats[label][column]["variance"]

                score += -0.5 * math.log(
                    2 * math.pi * variance
                )

                score += -(
                    (value - mean) ** 2
                ) / (
                    2 * variance
                )

            for column in CATEGORICAL_COLUMNS:

                value = str(row[column])

                score += (
                    self.category_log_probability[label][column]
                    .get(
                        value,
                        self.unknown_category_log_probability[label][column]
                    )
                )

            scores[label] = score

        return max(
            scores,
            key=scores.get
        )

    def predict(
        self,
        features
    ):

        return np.array(
            [
                self.predict_one(row)
                for _, row in features.iterrows()
            ]
        )
