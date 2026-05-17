"""
utils.py — logging, timer, và các helper functions dùng chung
"""

import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Any


# ── Logger setup ──────────────────────────────────────────────────────────────

def get_logger(name: str = "ray_exp") -> logging.Logger:
    """Trả về logger có format chuẩn."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Timer context manager ──────────────────────────────────────────────────────

class Timer:
    """
    Đo thời gian chạy của một block code.

    Dùng như:
        with Timer("ETL") as t:
            ...
        print(t.elapsed)   # giây
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0
        self._logger = get_logger()

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start
        if self.label:
            self._logger.info(f"{self.label} finished in {self.elapsed:.2f}s")


# ── Metrics I/O ───────────────────────────────────────────────────────────────

def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """
    Ghi metrics dict ra file JSON (tự tạo thư mục nếu chưa có).

    Args:
        metrics: dict chứa các chỉ số cần lưu.
        path:    đường dẫn file .json đầu ra.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Thêm timestamp để dễ trace
    metrics["saved_at"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

    get_logger().info(f"Metrics saved → {path}")


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Đọc metrics từ file JSON."""
    with open(path) as f:
        return json.load(f)


# ── Label helpers ──────────────────────────────────────────────────────────────

# AG News có 4 lớp, label gốc là 0-based sau khi load qua HuggingFace
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def label_to_name(label: int) -> str:
    return LABEL_NAMES.get(label, f"Unknown({label})")


# ── Throughput calculator ──────────────────────────────────────────────────────

def calc_throughput(n_samples: int, elapsed_sec: float) -> float:
    """Trả về samples/second, tránh chia cho 0."""
    if elapsed_sec <= 0:
        return float("inf")
    return n_samples / elapsed_sec