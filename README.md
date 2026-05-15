# Naive Bayes ашиглан донтолтын түвшин тогтоох нь

Энэхүү төсөл нь **Naive Bayes алгоритм** ашиглан Gen Z хэрэглэгчдийн social media хэрэглээний өгөгдлөөс **донтолтын түвшин** буюу `addiction_level`-ийг автоматаар ангилах машин сургалтын төсөл юм.

Загвар нь `Low`, `Medium`, `High` гэсэн гурван ангиллыг таамаглана.

Төслийн хүрээнд гаднын машин сургалтын сан ашиглалгүйгээр Python хэл дээр Naive Bayes ангилагчийг эхнээс нь хэрэгжүүлсэн.

---

## Төслийн зорилго

Энэхүү төслийн зорилго:

* Social media хэрэглээ болон донтолтын түвшний хамаарлыг машин сургалтаар судлах
* Naive Bayes алгоритмыг Python хэл дээр хэрэгжүүлэх
* Өгөгдлийг цэвэрлэж, машин сургалтад тохиромжтой хэлбэрт оруулах
* Тоон болон категори хувьсагчтай өгөгдөл дээр ангиллын загвар сургах
* Accuracy, Precision, Recall, F1-score, Confusion matrix ашиглан үр дүн үнэлэх
* Үр дүнг графикаар дүрслэх

---

## Гол хэрэгжүүлэлтүүд

Энэхүү төсөлд дараах үндсэн хэсгүүдийг хэрэгжүүлсэн:

* CSV өгөгдөл унших
* Missing value цэвэрлэх
* Train болон dev set болгон хуваах
* Gaussian Naive Bayes
* Categorical Naive Bayes
* Laplace smoothing
* Log probability ашигласан ангилалт
* Accuracy, Precision, Recall, F1-score
* Confusion matrix
* SVG visualization
* Quarto тайлан болон слайдын template

---

## Төслийн бүтэц

```text
Magadlal/
│
├── genz_social_media_usage_1M.csv
│
├── data/
│   └── processed/
│       ├── train.csv
│       ├── dev.csv
│       └── metadata.json
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── naive_bayes.py
│   ├── metrics.py
│   ├── visualization.py
│   └── main.py
│
├── models/
│   └── naive_bayes_model.pkl
│
├── outputs/
│   ├── metrics.json
│   ├── dev_predictions_sample.csv
│   └── figures/
│       ├── class_distribution.svg
│       ├── confusion_matrix.svg
│       └── numeric_feature_means.svg
│
├── report/
│   └── report.qmd
│
├── slides/
│   └── presentation.qmd
│
├── README.md
└── requirements.txt
```

---

## Өгөгдөл

Өгөгдөл нь `genz_social_media_usage_1M.csv` файл бөгөөд social media хэрэглээтэй холбоотой дараах хувьсагчдыг агуулна.

Тоон хувьсагчид:

* `age`
* `daily_usage_hours`
* `num_platforms_used`
* `avg_session_minutes`
* `night_usage`
* `mental_health_score`
* `screen_time_before_sleep`

Категори хувьсагчид:

* `gender`
* `country`
* `primary_platform`
* `purpose`

Таамаглах багана:

* `addiction_level`

Ангиллууд:

* `Low`
* `Medium`
* `High`

---

## Суулгах заавар

Virtual environment үүсгэх:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Шаардлагатай багцууд суулгах:

```bash
python3 -m pip install -r requirements.txt
```

macOS дээр `externally-managed-environment` гэсэн алдаа гарвал системийн Python руу шууд package суулгах гэж байна гэсэн үг. Дээрх шиг `.venv` үүсгээд идэвхжүүлсний дараа `pip install` ажиллуулна.

---

## Ажиллуулах

Өгөгдөл бэлтгэх, модел сургах, үнэлгээ хийх, visualization үүсгэх бүх алхмыг нэг дор ажиллуулах:

```bash
python3 main.py
```

Хүлээгдэх үр дүн:

```text
Accuracy: 93.05%
Macro F1: 92.29%
Confusion matrix:
High: [29212, 0, 2575]
Low: [0, 46808, 3436]
Medium: [3613, 4277, 110079]
```

---

## Загварын үр дүн

| Үзүүлэлт | Утга |
| --- | ---: |
| Accuracy | 93.05% |
| Macro F1 | 92.29% |

Дэлгэрэнгүй үр дүн:

* `outputs/metrics.json`
* `outputs/dev_predictions_sample.csv`
* `outputs/figures/confusion_matrix.svg`
* `outputs/figures/class_distribution.svg`
* `outputs/figures/numeric_feature_means.svg`

---

## Ашигласан аргачлал

Naive Bayes нь дараах томъёонд суурилна.

```text
P(C|X) = P(X|C) * P(C) / P(X)
```

Энд:

* `C` - ангилал буюу `Low`, `Medium`, `High`
* `X` - хэрэглэгчийн social media хэрэглээний шинжүүд

Тоон хувьсагчдад Gaussian Naive Bayes ашигласан. Категори хувьсагчдад Laplace smoothing ашиглан нөхцөлт магадлалыг тооцсон. Маш бага магадлал үржигдэх үед floating point алдаа гарахаас сэргийлж log probability ашигласан.

---

## Тайлан боловсруулах

Төслийн тайланг Quarto ашиглан PDF болгох:

```bash
quarto render report/report.qmd
```

RevealJS HTML слайд үүсгэх:

```bash
quarto render slides/presentation.qmd
```

PowerPoint файл үүсгэх:

```bash
quarto render slides/presentation.qmd --to pptx
```

Үүсэх файлууд:

* `slides/presentation.html`
* `slides/presentation.pptx`

---

## Ашиглалтын нөхцөл

Энэхүү төсөл нь сургалт, судалгааны зориулалтаар боловсруулагдсан.
