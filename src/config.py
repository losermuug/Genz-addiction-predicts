from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = ROOT_DIR / "genz_social_media_usage_1M.csv"
PROCESSED_DIR = ROOT_DIR / "data/processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
DEV_PATH = PROCESSED_DIR / "dev.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.json"

MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"

OUTPUT_DIR = ROOT_DIR / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "dev_predictions_sample.csv"
FIGURES_DIR = OUTPUT_DIR / "figures"

TARGET_COLUMN = "addiction_level"

NUMERIC_COLUMNS = [
    "age",
    "daily_usage_hours",
    "num_platforms_used",
    "avg_session_minutes",
    "night_usage",
    "mental_health_score",
    "screen_time_before_sleep",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "country",
    "primary_platform",
    "purpose",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
CLASS_LABELS = ["High", "Low", "Medium"]
