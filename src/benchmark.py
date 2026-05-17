"""
benchmark.py — Đo throughput, latency và xuất bảng so sánh framework

Dùng:
    from src.benchmark import run_benchmark, compare_frameworks
    stats = run_benchmark(train_ds, cfg)
    compare_frameworks("results/metrics")
"""

from __future__ import annotations

import os
import json
import time
import statistics
from pathlib import Path
from typing import Any

import ray
import ray.data

from src.utils import get_logger, save_metrics, Timer, calc_throughput

logger = get_logger(__name__)


# ── Đo throughput Ray Data ETL ─────────────────────────────────────────────────

def benchmark_etl(
    ds: ray.data.Dataset,
    n_warmup: int = 1,
    n_runs: int = 3,
) -> dict[str, float]:
    """
    Đo throughput của Ray Data pipeline qua nhiều lần chạy.

    Args:
        ds:       Ray Dataset đã tokenize.
        n_warmup: số lần warmup (không tính vào kết quả).
        n_runs:   số lần đo thực.

    Returns:
        dict gồm mean/median/std throughput (samples/sec) và latency (ms/sample).
    """
    logger.info(f"Benchmarking ETL — warmup={n_warmup}, runs={n_runs}")

    # Materialize trước để các lần đo sau không re-tokenize
    logger.info("Materializing dataset (tránh lazy re-execution khi benchmark)...")
    ds = ds.materialize()

    # Warmup
    for _ in range(n_warmup):
        ds.count()

    elapsed_list = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        n  = ds.count()
        elapsed_list.append(time.perf_counter() - t0)
        logger.info(f"  Run {i+1}/{n_runs}: {elapsed_list[-1]:.3f}s")

    throughputs = [calc_throughput(n, e) for e in elapsed_list]
    latencies   = [e / n * 1000 for e in elapsed_list]  # ms per sample

    stats = {
        "n_samples":          n,
        "throughput_mean":    statistics.mean(throughputs),
        "throughput_median":  statistics.median(throughputs),
        "throughput_std":     statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "latency_ms_mean":    statistics.mean(latencies),
        "n_runs":             n_runs,
    }

    logger.info(
        f"ETL throughput: {stats['throughput_mean']:.0f} ± "
        f"{stats['throughput_std']:.0f} samples/sec"
    )
    return stats


# ── Đo training throughput ─────────────────────────────────────────────────────

def benchmark_training(train_metrics_path: str) -> dict[str, Any]:
    """
    Đọc kết quả từ train.py và tính throughput trung bình per epoch.
    """
    if not os.path.exists(train_metrics_path):
        logger.warning(f"Train metrics not found: {train_metrics_path}")
        return {}

    with open(train_metrics_path) as f:
        m = json.load(f)

    return {
        "train_throughput":   m.get("throughput", 0),
        "total_train_time":   m.get("total_train_time", 0),
        "num_workers":        m.get("num_workers", 1),
        "final_train_acc":    m.get("train_acc", 0),
    }


# ── Full benchmark run ─────────────────────────────────────────────────────────

def run_benchmark(
    train_ds: ray.data.Dataset,
    cfg: dict,
    results_dir: str = "results",
) -> dict[str, Any]:
    """
    Chạy toàn bộ benchmark pipeline:
    1. ETL throughput
    2. Training throughput (từ file metrics đã lưu)
    3. Gộp và lưu kết quả

    Returns:
        dict tổng hợp tất cả benchmark metrics.
    """
    logger.info("=== Running full benchmark ===")

    etl_stats  = benchmark_etl(train_ds)
    train_path = os.path.join(results_dir, "metrics", "ray_train_metrics.json")
    train_stats = benchmark_training(train_path)

    combined = {
        "framework": "Ray",
        **etl_stats,
        **train_stats,
    }

    out_path = os.path.join(results_dir, "metrics", "benchmark_ray.json")
    save_metrics(combined, out_path)

    return combined


# ── So sánh nhiều framework ────────────────────────────────────────────────────

def compare_frameworks(metrics_dir: str = "results/metrics") -> None:
    """
    Đọc tất cả file benchmark_*.json và in bảng so sánh.
    Dùng sau khi đã chạy xong experiments của cả 4 framework.
    """
    metrics_dir = Path(metrics_dir)
    files = sorted(metrics_dir.glob("benchmark_*.json"))

    if not files:
        logger.warning("No benchmark files found. Run experiments first.")
        return

    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        rows.append({
            "Framework":          d.get("framework", fp.stem),
            "ETL throughput":     f"{d.get('throughput_mean', 0):.0f} s/s",
            "Train throughput":   f"{d.get('train_throughput', 0):.0f} s/s",
            "Train time (s)":     f"{d.get('total_train_time', 0):.1f}",
            "Workers":            d.get("num_workers", "-"),
            "Train acc":          f"{d.get('final_train_acc', 0):.4f}",
        })

    # In bảng đơn giản
    if rows:
        headers = list(rows[0].keys())
        col_w   = [max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in headers]

        sep  = "+" + "+".join("-" * w for w in col_w) + "+"
        head = "|" + "|".join(h.center(w) for h, w in zip(headers, col_w)) + "|"

        print(sep)
        print(head)
        print(sep)
        for r in rows:
            print("|" + "|".join(str(r[h]).center(w) for h, w in zip(headers, col_w)) + "|")
        print(sep)