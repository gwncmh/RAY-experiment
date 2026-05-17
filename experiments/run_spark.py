"""
experiments/run_spark.py — Apache Spark baseline cho AG News text classification

Dùng TF-IDF + LogisticRegression (MLlib) làm baseline truyền thống,
và Spark NLP (nếu cài được) cho deep learning pipeline.

Chạy:
    python experiments/run_spark.py
    python experiments/run_spark.py --config configs/spark_config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, save_metrics, Timer

logger = get_logger("run_spark")


# ── Kiểm tra PySpark ───────────────────────────────────────────────────────────

def _check_pyspark():
    try:
        import pyspark  # noqa: F401
        return True
    except ImportError:
        logger.error("PySpark chưa cài. Chạy: pip install pyspark")
        return False


# ── Load AG News → Spark DataFrame ────────────────────────────────────────────

def load_spark_datasets(spark, cfg: dict):
    """
    Load AG News từ HuggingFace rồi chuyển sang Spark DataFrame.
    """
    from datasets import load_dataset

    logger.info("Loading AG News...")
    hf = load_dataset("ag_news", cache_dir=cfg.get("data_cache"))

    # Chuyển sang pandas trước, rồi spark (cách đơn giản nhất)
    train_pd = hf["train"].to_pandas()
    test_pd  = hf["test"].to_pandas()

    train_sdf = spark.createDataFrame(train_pd)
    test_sdf  = spark.createDataFrame(test_pd)

    logger.info(
        f"Spark DataFrames created — "
        f"train: {train_sdf.count()}, test: {test_sdf.count()}"
    )
    return train_sdf, test_sdf


# ── TF-IDF + Logistic Regression pipeline ─────────────────────────────────────

def build_tfidf_pipeline(cfg: dict):
    """
    Spark MLlib pipeline: Tokenizer → HashingTF → IDF → LogisticRegression
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashing_tf = HashingTF(
        inputCol="words",
        outputCol="raw_features",
        numFeatures=cfg.get("num_features", 65536),
    )
    idf = IDF(inputCol="raw_features", outputCol="features", minDocFreq=5)
    lr  = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=cfg.get("max_iter", 100),
        regParam=cfg.get("reg_param", 0.01),
        elasticNetParam=0.0,
        family="multinomial",
    )

    return Pipeline(stages=[tokenizer, hashing_tf, idf, lr])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_spark(predictions, cfg: dict, results_dir: str) -> dict:
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    accuracy = evaluator_acc.evaluate(predictions)
    f1_macro = evaluator_f1.evaluate(predictions)

    metrics = {
        "framework":   "Spark",
        "model":       "TF-IDF + LogisticRegression",
        "accuracy":    accuracy,
        "f1_macro":    f1_macro,
    }

    out_path = os.path.join(results_dir, "metrics", "eval_spark.json")
    save_metrics(metrics, out_path)

    logger.info(f"Spark | Accuracy: {accuracy:.4f} | F1: {f1_macro:.4f}")
    return metrics


# ── Benchmark ETL throughput ───────────────────────────────────────────────────

def benchmark_spark_etl(train_sdf, n_runs: int = 3) -> dict:
    """Đo throughput Spark ETL qua nhiều lần count()."""
    from pyspark.ml.feature import Tokenizer, HashingTF

    tokenizer  = Tokenizer(inputCol="text", outputCol="words")
    hashing_tf = HashingTF(inputCol="words", outputCol="features", numFeatures=65536)

    # Warmup
    tokenizer.transform(train_sdf).count()

    times = []
    n = 0
    for i in range(n_runs):
        t0 = time.perf_counter()
        transformed = hashing_tf.transform(tokenizer.transform(train_sdf))
        n = transformed.count()
        times.append(time.perf_counter() - t0)
        logger.info(f"  ETL run {i+1}/{n_runs}: {times[-1]:.3f}s")

    import statistics
    throughputs = [n / t for t in times]

    return {
        "framework":         "Spark",
        "n_samples":         n,
        "throughput_mean":   statistics.mean(throughputs),
        "throughput_median": statistics.median(throughputs),
        "throughput_std":    statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
        "latency_ms_mean":   statistics.mean(t / n * 1000 for t in times),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not _check_pyspark():
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Spark AG News baseline")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    DEFAULT_CFG = {
        "num_features": 65536,
        "max_iter":     100,
        "reg_param":    0.01,
        "data_cache":   None,
        "results_dir":  "results",
        "num_runs":     3,
    }
    cfg = DEFAULT_CFG.copy()
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("ag_news_spark_baseline")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info(f"Spark version: {spark.version}")

    # 1. Load data
    train_sdf, test_sdf = load_spark_datasets(spark, cfg)

    # 2. Benchmark ETL
    logger.info("=" * 50)
    logger.info("Benchmarking Spark ETL")
    logger.info("=" * 50)
    etl_stats = benchmark_spark_etl(train_sdf, n_runs=cfg["num_runs"])

    bench_path = os.path.join(cfg["results_dir"], "metrics", "benchmark_spark.json")
    save_metrics(etl_stats, bench_path)
    logger.info(f"Spark ETL throughput: {etl_stats['throughput_mean']:.0f} samples/sec")

    # 3. Train TF-IDF pipeline
    logger.info("=" * 50)
    logger.info("Training TF-IDF + LogisticRegression")
    logger.info("=" * 50)
    pipeline = build_tfidf_pipeline(cfg)

    with Timer("Spark training") as t:
        model = pipeline.fit(train_sdf)

    train_time = t.elapsed

    # 4. Evaluate
    predictions = model.transform(test_sdf)
    eval_metrics = evaluate_spark(predictions, cfg, cfg["results_dir"])

    # Cập nhật benchmark với train time
    etl_stats["total_train_time"] = train_time
    etl_stats["train_throughput"] = train_sdf.count() / max(train_time, 1)
    save_metrics(etl_stats, bench_path)

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY — Spark")
    logger.info("=" * 50)
    logger.info(f"  ETL throughput : {etl_stats['throughput_mean']:.0f} samples/sec")
    logger.info(f"  Training time  : {train_time:.1f}s")
    logger.info(f"  Accuracy       : {eval_metrics['accuracy']:.4f}")
    logger.info(f"  F1 macro       : {eval_metrics['f1_macro']:.4f}")

    spark.stop()


if __name__ == "__main__":
    main()