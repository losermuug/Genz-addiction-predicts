from html import escape

import pandas as pd

from src.config import FIGURES_DIR
from src.config import NUMERIC_COLUMNS
from src.config import TARGET_COLUMN


CLASS_COLORS = {
    "Low": "#2f80ed",
    "Medium": "#27ae60",
    "High": "#eb5757",
}


def svg_document(
    width,
    height,
    body
):

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        '<style>'
        'text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
        '.title{font-size:22px;font-weight:700}'
        '.label{font-size:12px}'
        '.small{font-size:11px;fill:#555}'
        '</style>\n'
        f'{body}\n'
        '</svg>\n'
    )


def save_class_distribution(
    data,
    output_path
):

    counts = (
        data[TARGET_COLUMN]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
    )

    width = 760
    height = 460
    chart_x = 90
    chart_y = 80
    chart_h = 290
    max_count = max(
        int(counts.max()),
        1
    )

    body = [
        '<text x="40" y="42" class="title">'
        'Addiction level class distribution'
        '</text>'
    ]

    body.append(
        f'<line x1="{chart_x}" y1="{chart_y + chart_h}" '
        f'x2="680" y2="{chart_y + chart_h}" stroke="#333"/>'
    )

    body.append(
        f'<line x1="{chart_x}" y1="{chart_y}" '
        f'x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#333"/>'
    )

    for index, (label, count) in enumerate(counts.items()):

        bar_width = 115
        bar_height = int(
            count / max_count * chart_h
        )

        x = chart_x + 70 + index * 195
        y = chart_y + chart_h - bar_height

        body.append(
            f'<rect x="{x}" y="{y}" '
            f'width="{bar_width}" height="{bar_height}" '
            f'fill="{CLASS_COLORS[label]}"/>'
        )

        body.append(
            f'<text x="{x + bar_width / 2}" y="{chart_y + chart_h + 26}" '
            f'text-anchor="middle" class="label">{escape(label)}</text>'
        )

        body.append(
            f'<text x="{x + bar_width / 2}" y="{y - 8}" '
            f'text-anchor="middle" class="label">{int(count):,}</text>'
        )

    output_path.write_text(
        svg_document(
            width,
            height,
            "\n".join(body)
        ),
        encoding="utf-8"
    )


def save_confusion_matrix(
    report,
    output_path
):

    labels = report["confusion_matrix"]["labels"]
    matrix = report["confusion_matrix"]["matrix"]

    width = 680
    height = 560
    cell = 110
    start_x = 190
    start_y = 120
    max_value = max(
        max(row)
        for row in matrix
    ) or 1

    body = [
        '<text x="40" y="42" class="title">'
        'Naive Bayes confusion matrix'
        '</text>',
        '<text x="330" y="90" text-anchor="middle" class="label">'
        'Predicted label'
        '</text>',
    ]

    for index, label in enumerate(labels):

        body.append(
            f'<text x="{start_x + index * cell + cell / 2}" '
            f'y="{start_y - 18}" text-anchor="middle" '
            f'class="label">{escape(label)}</text>'
        )

        body.append(
            f'<text x="{start_x - 18}" '
            f'y="{start_y + index * cell + cell / 2 + 4}" '
            f'text-anchor="end" class="label">{escape(label)}</text>'
        )

    for row_index, row in enumerate(matrix):

        for column_index, value in enumerate(row):

            intensity = value / max_value
            red = int(245 - intensity * 160)
            green = int(248 - intensity * 130)
            blue = int(255 - intensity * 35)

            x = start_x + column_index * cell
            y = start_y + row_index * cell

            body.append(
                f'<rect x="{x}" y="{y}" width="{cell}" '
                f'height="{cell}" fill="#{red:02x}{green:02x}{blue:02x}" '
                f'stroke="#ffffff" stroke-width="2"/>'
            )

            body.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 5}" '
                f'text-anchor="middle" class="label">{int(value):,}</text>'
            )

    output_path.write_text(
        svg_document(
            width,
            height,
            "\n".join(body)
        ),
        encoding="utf-8"
    )


def save_numeric_feature_means(
    data,
    output_path
):

    means = (
        data
        .groupby(TARGET_COLUMN)[NUMERIC_COLUMNS]
        .mean()
        .reindex(["Low", "Medium", "High"])
    )

    normalized = means.copy()

    for column in NUMERIC_COLUMNS:

        minimum = means[column].min()
        maximum = means[column].max()

        if maximum == minimum:
            normalized[column] = 0.5
        else:
            normalized[column] = (
                means[column]
                -
                minimum
            ) / (
                maximum
                -
                minimum
            )

    width = 960
    height = 560
    chart_x = 110
    chart_y = 95
    chart_h = 330
    group_width = 780 / len(NUMERIC_COLUMNS)

    body = [
        '<text x="40" y="42" class="title">'
        'Numeric feature means by addiction level'
        '</text>',
        f'<line x1="{chart_x}" y1="{chart_y + chart_h}" '
        f'x2="890" y2="{chart_y + chart_h}" stroke="#333"/>',
        f'<line x1="{chart_x}" y1="{chart_y}" '
        f'x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#333"/>',
    ]

    for feature_index, column in enumerate(NUMERIC_COLUMNS):

        group_x = chart_x + feature_index * group_width + 28

        for class_index, label in enumerate(["Low", "Medium", "High"]):

            value = float(
                normalized.loc[label, column]
            )

            bar_height = int(
                value * chart_h
            )

            x = group_x + class_index * 26
            y = chart_y + chart_h - bar_height

            body.append(
                f'<rect x="{x}" y="{y}" width="22" '
                f'height="{bar_height}" fill="{CLASS_COLORS[label]}"/>'
            )

        body.append(
            f'<text x="{group_x + 36}" y="{chart_y + chart_h + 28}" '
            f'text-anchor="middle" class="small">'
            f'{escape(column.replace("_", " "))}</text>'
        )

    output_path.write_text(
        svg_document(
            width,
            height,
            "\n".join(body)
        ),
        encoding="utf-8"
    )


def create_visualizations(
    train_data,
    report
):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_class_distribution(
        train_data,
        FIGURES_DIR / "class_distribution.svg"
    )

    save_confusion_matrix(
        report,
        FIGURES_DIR / "confusion_matrix.svg"
    )

    save_numeric_feature_means(
        train_data,
        FIGURES_DIR / "numeric_feature_means.svg"
    )
