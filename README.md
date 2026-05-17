# Text Classification — AG News Benchmark

Nghiên cứu so sánh 4 framework xử lý Big Data cho bài toán phân loại văn bản quy mô lớn.

| Framework | Vai trò | Dataset |
|-----------|---------|---------|
| **Ray**     | Distributed training (fine-tune BERT) | AG News 120k |
| **Spark**   | TF-IDF + LR baseline                  | AG News 120k |
| **MindsDB** | AI-native SQL interface               | AG News 120k |
| **vLLM**    | Optimized LLM inference               | AG News 7.6k |

---

## Cài đặt

```bash
# 1. Clone / tải project
cd RAY-experiment

# 2. Tạo virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 3. Cài dependencies
pip install -r requirements.txt
```

---

## Cấu trúc project

```
ray-text-classification/
├── src/                    # Logic chính — import từ đây
│   ├── utils.py            # Logger, Timer, save_metrics
│   ├── data_pipeline.py    # Load AG News → Ray Dataset → tokenize
│   ├── model.py            # BERT classifier
│   ├── train.py            # Ray Train distributed loop
│   ├── evaluate.py         # Accuracy, F1, confusion matrix
│   └── benchmark.py        # Throughput benchmark, bảng so sánh
│
├── experiments/            # Entry point chạy từng framework
│   ├── run_ray.py          # ← Chạy cái này trước
│   ├── run_spark.py        # Spark baseline
│   ├── run_mindsdb.py      # MindsDB (cần server riêng)
│   └── run_vllm.py         # vLLM (cần GPU)
│
├── configs/                # Hyperparameters tách khỏi code
│   ├── ray_config.yaml
│   └── spark_config.yaml
│
├── notebooks/              # EDA và phân tích kết quả
│   ├── 01_eda.ipynb        # Khám phá AG News
│   ├── 02_prototype.ipynb  # Test pipeline trên tập nhỏ
│   └── 03_analysis.ipynb   # Vẽ biểu đồ cho báo cáo
│
├── results/
│   ├── metrics/            # JSON files kết quả thực nghiệm
│   └── figures/            # Biểu đồ .png dùng trong báo cáo
│
└── data/
    └── raw/                # CSV files (gitignored)
```

---

## Thứ tự chạy

### Bước 1 — EDA (tùy chọn nhưng nên làm)
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Bước 2 — Prototype nhanh (kiểm tra pipeline)
```bash
jupyter notebook notebooks/02_prototype.ipynb
```

### Bước 3 — Chạy Ray (framework chính)
```bash
python experiments/run_ray.py --config configs/ray_config.yaml
```

Kết quả lưu tại:
- `results/metrics/ray_train_metrics.json`
- `results/metrics/eval_metrics.json`
- `results/metrics/benchmark_ray.json`

### Bước 4 — Chạy Spark baseline
```bash
pip install pyspark
python experiments/run_spark.py
```

### Bước 5 — Chạy MindsDB (cần Docker)
```bash
docker run -p 47334:47334 -p 47335:47335 mindsdb/mindsdb
python experiments/run_mindsdb.py
```

### Bước 6 — Chạy vLLM (cần GPU + Linux)
```bash
pip install vllm
python experiments/run_vllm.py --model facebook/opt-1.3b --n-samples 500
```

> Nếu không có GPU, `run_vllm.py` tự động lưu số liệu từ literature review.

### Bước 7 — Phân tích kết quả
```bash
jupyter notebook notebooks/03_analysis.ipynb
```

So sánh tất cả framework:
```python
from src.benchmark import compare_frameworks
compare_frameworks("results/metrics")
```

---

## Tùy chỉnh config

Chỉnh `configs/ray_config.yaml` để thay đổi hyperparameters mà không cần sửa code:

```yaml
model: bert-base-uncased   # đổi thành distilbert-base-uncased để nhanh hơn
epochs: 3
batch_size: 32
num_workers: 2             # tăng nếu có nhiều CPU
use_gpu: false             # true nếu có GPU
```

---

## Kết quả mong đợi

| Framework | ETL (samples/s) | Accuracy | F1 Macro | Train time |
|-----------|-----------------|----------|----------|------------|
| Ray       | ~8,000–15,000   | ~0.94    | ~0.94    | ~20 min    |
| Spark     | ~3,000–6,000    | ~0.91    | ~0.91    | ~5 min     |
| MindsDB   | N/A (SQL)       | ~0.90    | ~0.90    | ~15 min    |
| vLLM      | ~2,000–5,000*   | ~0.89    | ~0.88    | 0 (zero-shot) |

*tokens/sec, inference only

---

## Tài liệu tham khảo chính

- Moritz et al. (2018) — Ray: A Distributed Framework for Emerging AI Applications
- Kwon et al. (2023) — Efficient Memory Management for LLM Serving with PagedAttention
- Zaharia et al. (2016) — Apache Spark: A Unified Engine for Big Data Processing
- Zhang et al. (2015) — Character-level Convolutional Networks for Text Classification (AG News)