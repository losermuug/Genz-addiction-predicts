import json
import pickle
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT_DIR)
    )

from src.config import FEATURE_COLUMNS
from src.config import METRICS_PATH
from src.config import MODEL_DIR
from src.config import MODEL_PATH
from src.config import OUTPUT_DIR
from src.config import PREDICTIONS_PATH
from src.config import TARGET_COLUMN
from src.dataset import read_dataset
from src.metrics import classification_report
from src.naive_bayes import NaiveBayes
from src.visualization import create_visualizations


train_data, dev_data = read_dataset()

train_features = train_data[FEATURE_COLUMNS]
train_labels = train_data[TARGET_COLUMN].astype(str)

dev_features = dev_data[FEATURE_COLUMNS]
dev_labels = dev_data[TARGET_COLUMN].astype(str)


model = NaiveBayes(
    alpha=1
)

model.fit(
    train_features,
    train_labels
)


predictions = model.predict(
    dev_features
)


report = classification_report(
    predictions,
    dev_labels,
    model.classes
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with MODEL_PATH.open("wb") as file:
    pickle.dump(
        model,
        file
    )

METRICS_PATH.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

prediction_sample = dev_data.copy()
prediction_sample["predicted_addiction_level"] = predictions
prediction_sample.head(1000).to_csv(
    PREDICTIONS_PATH,
    index=False
)

create_visualizations(
    train_data,
    report
)


print(
    f"Accuracy: {report['accuracy']:.2%}"
)

print(
    f"Macro F1: {report['macro_f1']:.2%}"
)

print(
    "Confusion matrix:"
)

for label, row in zip(
    report["confusion_matrix"]["labels"],
    report["confusion_matrix"]["matrix"]
):
    print(
        f"{label}: {row}"
    )
