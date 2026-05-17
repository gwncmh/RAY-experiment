"""
experiments/run_ray.py — Entry point chạy toàn bộ pipeline Ray

Chạy:
    python experiments/run_ray.py
    python experiments/run_ray.py --config configs/ray_config.yaml
    python experiments/run_ray.py --config configs/ray_config.yaml --skip-train
"""

from __future__ import annotations

import argparse
import os
import sys

# Thêm root vào sys.path để import src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import ray

from src import (
    build_datasets,
    build_model,
    run_training,
    evaluate_model,
    run_benchmark,
)
from src.model import load_checkpoint
from src.utils import get_logger
from src.benchmark import benchmark_etl
from src.utils import save_metrics

logger = get_logger("run_ray")


# ── Default config (dùng khi không có file yaml) ───────────────────────────────

DEFAULT_CFG = {
    "model":          "bert-base-uncased",
    "num_labels":     4,
    "max_length":     128,
    "batch_size":     32,
    "eval_batch_size": 64,
    "lr":             2e-5,
    "weight_decay":   0.01,
    "epochs":         3,
    "num_workers":    2,
    "use_gpu":        False,
    "data_cache":     None,
    "results_dir":    "results",
    "framework":      "ray",
}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ray AG News classification pipeline")
    parser.add_argument("--config",     default=None,    help="Path to yaml config")
    parser.add_argument("--skip-train", action="store_true", help="Bỏ qua training, chỉ evaluate")
    parser.add_argument("--checkpoint", default=None,    help="Load checkpoint có sẵn")
    args = parser.parse_args()

    # ── Load config ──
    cfg = DEFAULT_CFG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))
        logger.info(f"Config loaded from {args.config}")
    else:
        logger.info("Using default config")

    # ── Init Ray ──
    ray.init(ignore_reinit_error=True)
    logger.info(f"Ray cluster resources: {ray.cluster_resources()}")

    # ── Step 1: Load & tokenize dataset ──
    logger.info("=" * 50)
    logger.info("STEP 1 — Data pipeline")
    logger.info("=" * 50)
    train_ds, test_ds = build_datasets(cfg)

    # ── Step 2: Benchmark ETL (chỉ đo throughput ETL, chưa merge train metrics) ──
    logger.info("=" * 50)
    logger.info("STEP 2 — Benchmark ETL throughput")
    logger.info("=" * 50)
    etl_stats = benchmark_etl(train_ds)
    etl_stats["framework"] = "Ray"
    bench = etl_stats  # dùng tạm, sẽ update sau khi train xong
    logger.info(f"ETL throughput: {bench['throughput_mean']:.0f} samples/sec")

    # ── Step 3: Train (nếu không skip) ──
    if not args.skip_train:
        logger.info("=" * 50)
        logger.info("STEP 3 — Distributed training")
        logger.info("=" * 50)
        train_metrics = run_training(train_ds, cfg, results_dir=cfg["results_dir"])
        logger.info(f"Training complete: {train_metrics}")
    
        # Merge train metrics vào benchmark SAU KHI train xong
        bench["total_train_time"] = train_metrics.get("total_train_time", 0)
        bench["train_throughput"] = train_metrics.get("throughput", 0)
        bench["num_workers"]      = train_metrics.get("num_workers", cfg.get("num_workers", 1))
        bench["final_train_acc"]  = train_metrics.get("train_acc", 0)
        bench_path = os.path.join(cfg["results_dir"], "metrics", "benchmark_ray.json")
        save_metrics(bench, bench_path)
    else:
        logger.info("STEP 3 — Skipped (--skip-train)")

    # ── Step 4: Evaluate ──
    logger.info("=" * 50)
    logger.info("STEP 4 — Evaluation on test set")
    logger.info("=" * 50)

    # Load checkpoint từ Ray Train result (nếu vừa train xong)
    model = build_model(cfg["model"], cfg["num_labels"])
    if not args.skip_train and 'train_metrics' in dir():
        # Ray Train lưu checkpoint vào thư mục ray_results/
        # Tìm checkpoint mới nhất
        import glob
        ckpt_dirs = sorted(
            glob.glob("ray_results/ray_ag_news/**/checkpoint_*/", recursive=True)
        )
        if ckpt_dirs:
            latest_ckpt = ckpt_dirs[-1]
            logger.info(f"Loading Ray checkpoint: {latest_ckpt}")
            model = load_checkpoint(model, os.path.join(latest_ckpt, "model.pt"))
        else:
            logger.warning(
                "Không tìm thấy checkpoint Ray Train. "
                "Evaluate sẽ dùng pretrained weights (chưa fine-tune). "
                "Thêm torch.save() vào _train_loop để lưu model.pt."
            )
    elif args.checkpoint:
        model = load_checkpoint(model, args.checkpoint)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    eval_metrics = evaluate_model(model, test_ds, cfg, results_dir=cfg["results_dir"])

    # ── Summary ──
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"  Framework    : Ray")
    logger.info(f"  Model        : {cfg['model']}")
    logger.info(f"  Workers      : {cfg['num_workers']}")
    logger.info(f"  ETL thruput  : {bench['throughput_mean']:.0f} samples/sec")
    logger.info(f"  Accuracy     : {eval_metrics['accuracy']:.4f}")
    logger.info(f"  F1 macro     : {eval_metrics['f1_macro']:.4f}")

    ray.shutdown()


if __name__ == "__main__":
    main()