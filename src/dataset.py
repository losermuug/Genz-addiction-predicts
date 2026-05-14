import json
from pathlib import Path

import pandas as pd

from src.config import CATEGORICAL_COLUMNS
from src.config import DEV_PATH
from src.config import METADATA_PATH
from src.config import NUMERIC_COLUMNS
from src.config import PROCESSED_DIR
from src.config import RAW_DATA_PATH
from src.config import TARGET_COLUMN
from src.config import TRAIN_PATH


REQUIRED_COLUMNS = (
    NUMERIC_COLUMNS
    +
    CATEGORICAL_COLUMNS
    +
    [TARGET_COLUMN]
)


def read_raw_dataset(
    csv_path=RAW_DATA_PATH
):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}"
        )

    data = pd.read_csv(csv_path)

    missing_columns = sorted(
        set(REQUIRED_COLUMNS)
        -
        set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return data[REQUIRED_COLUMNS].copy()


def clean_dataset(data):

    missing_before = data.isna().sum().to_dict()
    row_count_before = len(data)

    for column in CATEGORICAL_COLUMNS + [TARGET_COLUMN]:
        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "nan": pd.NA,
                    "None": pd.NA,
                }
            )
        )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=[TARGET_COLUMN]
    )

    data = data[
        data[TARGET_COLUMN].isin(
            ["Low", "Medium", "High"]
        )
    ].copy()

    for column in NUMERIC_COLUMNS:
        data[column] = data[column].fillna(
            data[column].median()
        )

    for column in CATEGORICAL_COLUMNS:
        data[column] = (
            data[column]
            .fillna("Unknown")
            .astype(str)
        )

    data["age"] = data["age"].clip(
        lower=10,
        upper=35
    )

    data["daily_usage_hours"] = data[
        "daily_usage_hours"
    ].clip(
        lower=0,
        upper=24
    )

    data["avg_session_minutes"] = data[
        "avg_session_minutes"
    ].clip(
        lower=0
    )

    data["screen_time_before_sleep"] = data[
        "screen_time_before_sleep"
    ].clip(
        lower=0
    )

    data["mental_health_score"] = data[
        "mental_health_score"
    ].clip(
        lower=0,
        upper=10
    )

    data["night_usage"] = (
        data["night_usage"]
        .round()
        .clip(
            lower=0,
            upper=1
        )
    )

    data["num_platforms_used"] = (
        data["num_platforms_used"]
        .round()
        .clip(
            lower=1
        )
    )

    row_count_after_cleaning = len(data)
    data = data.drop_duplicates().reset_index(
        drop=True
    )

    summary = {
        "rows_before": int(row_count_before),
        "rows_after_cleaning": int(row_count_after_cleaning),
        "rows_after_deduplication": int(len(data)),
        "duplicates_removed": int(row_count_after_cleaning - len(data)),
        "missing_values_before_cleaning": {
            key: int(value)
            for key, value in missing_before.items()
        },
        "class_counts": {
            key: int(value)
            for key, value
            in data[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
    }

    return data, summary


def split_dataset(
    data,
    test_size=0.2,
    random_state=42
):

    dev_parts = []

    for _, class_rows in data.groupby(
        TARGET_COLUMN,
        sort=False
    ):
        dev_parts.append(
            class_rows.sample(
                frac=test_size,
                random_state=random_state
            )
        )

    dev_data = (
        pd.concat(dev_parts)
        .sample(
            frac=1,
            random_state=random_state
        )
    )

    train_data = (
        data.drop(dev_data.index)
        .sample(
            frac=1,
            random_state=random_state
        )
    )

    return (
        train_data.reset_index(drop=True),
        dev_data.reset_index(drop=True),
    )


def save_dataset(
    train_data,
    dev_data,
    summary,
    output_dir=PROCESSED_DIR
):

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_data.to_csv(
        TRAIN_PATH,
        index=False
    )

    dev_data.to_csv(
        DEV_PATH,
        index=False
    )

    metadata = {
        "source_file": str(RAW_DATA_PATH),
        "target_column": TARGET_COLUMN,
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "train_rows": int(len(train_data)),
        "dev_rows": int(len(dev_data)),
        "train_class_counts": {
            key: int(value)
            for key, value
            in train_data[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
        "dev_class_counts": {
            key: int(value)
            for key, value
            in dev_data[TARGET_COLUMN]
            .value_counts()
            .sort_index()
            .items()
        },
        "cleaning_summary": summary,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def prepare_dataset():

    raw_data = read_raw_dataset()
    clean_data, summary = clean_dataset(raw_data)

    train_data, dev_data = split_dataset(
        clean_data
    )

    save_dataset(
        train_data,
        dev_data,
        summary
    )

    return train_data, dev_data


def read_dataset():

    if (
        not TRAIN_PATH.exists()
        or
        not DEV_PATH.exists()
    ):
        return prepare_dataset()

    train_data = pd.read_csv(TRAIN_PATH)
    dev_data = pd.read_csv(DEV_PATH)

    return train_data, dev_data
