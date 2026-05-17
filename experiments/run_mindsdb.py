"""
experiments/run_mindsdb.py — MindsDB text classification qua SQL interface

MindsDB cho phép tạo model bằng SQL thuần:
    CREATE MODEL ag_classifier
    PREDICT label
    USING engine = 'huggingface', ...

Chạy:
    # Option A: MindsDB local (Docker)
    docker run -p 47334:47334 -p 47335:47335 mindsdb/mindsdb

    # Option B: MindsDB Cloud — đăng ký tại cloud.mindsdb.com

    python experiments/run_mindsdb.py --host 127.0.0.1 --port 47334
    python experiments/run_mindsdb.py --cloud --email YOUR_EMAIL --password YOUR_PASS
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import get_logger, save_metrics, Timer

logger = get_logger("run_mindsdb")


# ── Kết nối MindsDB ────────────────────────────────────────────────────────────

def connect_mindsdb(host: str = "127.0.0.1", port: int = 47334):
    """
    Kết nối tới MindsDB qua MySQL-compatible connector.
    MindsDB expose MySQL wire protocol trên port 47334.
    """
    try:
        import mindsdb_sdk
        server = mindsdb_sdk.connect(f"http://{host}:{port}")
        logger.info(f"Connected to MindsDB at {host}:{port}")
        return server
    except ImportError:
        logger.error("Cài MindsDB SDK: pip install mindsdb_sdk")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Kết nối MindsDB thất bại: {e}")
        logger.info("Hãy chắc MindsDB đang chạy: docker run -p 47334:47334 mindsdb/mindsdb")
        sys.exit(1)


# ── Upload AG News vào MindsDB ─────────────────────────────────────────────────

def upload_ag_news(server, cfg: dict) -> str:
    """
    Load AG News và đẩy vào MindsDB dưới dạng file CSV.
    Trả về tên table đã tạo.
    """
    from datasets import load_dataset
    import pandas as pd

    logger.info("Loading AG News...")
    hf = load_dataset("ag_news", cache_dir=cfg.get("data_cache"))

    train_pd = hf["train"].to_pandas()
    test_pd  = hf["test"].to_pandas()

    # Lưu ra CSV tạm
    os.makedirs("data/raw", exist_ok=True)
    train_path = "data/raw/ag_news_train.csv"
    test_path  = "data/raw/ag_news_test.csv"

    train_pd.to_csv(train_path, index=False)
    test_pd.to_csv(test_path,  index=False)

    logger.info(f"Saved: {train_path} ({len(train_pd)} rows)")
    logger.info(f"Saved: {test_path}  ({len(test_pd)} rows)")

    return train_path, test_path, len(train_pd), len(test_pd)


# ── Tạo MindsDB model bằng SQL ─────────────────────────────────────────────────

def create_mindsdb_model(server, cfg: dict) -> None:
    """
    Tạo text classification model trong MindsDB qua SQL.
    MindsDB hỗ trợ nhiều engine: huggingface, lightwood, openai.
    """
    model_name = cfg.get("model_name", "ag_news_classifier")
    engine     = cfg.get("engine", "huggingface")
    hf_model   = cfg.get("model", "distilbert-base-uncased")

    # Xoá model cũ nếu tồn tại
    try:
        server.query(f"DROP MODEL IF EXISTS {model_name};")
        logger.info(f"Dropped existing model: {model_name}")
    except Exception:
        pass

    if engine == "huggingface":
        create_sql = f"""
        CREATE MODEL {model_name}
        PREDICT label
        USING
            engine = 'huggingface',
            task = 'text-classification',
            model_name = '{hf_model}',
            input_column = 'text',
            labels = ['World', 'Sports', 'Business', 'Sci/Tech'];
        """
    elif engine == "lightwood":
        # Lightwood = MindsDB's native AutoML engine
        create_sql = f"""
        CREATE MODEL {model_name}
        FROM files (SELECT text, label FROM ag_news_train)
        PREDICT label
        USING engine = 'lightwood';
        """
    else:
        raise ValueError(f"Unknown engine: {engine}")

    logger.info(f"Creating MindsDB model: {model_name} (engine={engine})")
    logger.info(f"SQL:\n{create_sql.strip()}")

    t0 = time.perf_counter()
    server.query(create_sql)

    # Chờ model training xong (MindsDB train async)
    import time as time_mod
    max_wait = 600  # 10 phút
    start    = time_mod.time()

    while True:
        status_df = server.query(
            f"SELECT status FROM information_schema.models "
            f"WHERE name = '{model_name}';"
        ).fetch()

        if status_df.empty:
            logger.warning("Model not found yet, waiting...")
        else:
            status = status_df.iloc[0]["status"]
            logger.info(f"Model status: {status}")
            if status == "complete":
                break
            elif status == "error":
                logger.error("MindsDB model training failed!")
                break

        if time_mod.time() - start > max_wait:
            logger.error("Timeout waiting for MindsDB model")
            break

        time_mod.sleep(10)

    elapsed = time.perf_counter() - t0
    logger.info(f"Model ready in {elapsed:.1f}s")
    return elapsed


# ── Predict và đánh giá ────────────────────────────────────────────────────────

def evaluate_mindsdb(server, cfg: dict, n_test: int, results_dir: str) -> dict:
    """
    Dùng MindsDB SQL để predict và tính accuracy.
    """
    model_name = cfg.get("model_name", "ag_news_classifier")

    predict_sql = f"""
    SELECT t.label AS true_label, m.label AS pred_label
    FROM files.ag_news_test AS t
    JOIN {model_name} AS m;
    """

    logger.info("Running predictions via MindsDB SQL JOIN...")

    with Timer("MindsDB inference") as t:
        results = server.query(predict_sql).fetch()

    correct  = (results["true_label"] == results["pred_label"]).sum()
    accuracy = correct / len(results)

    # F1 bằng sklearn (tính ngoài MindsDB)
    from sklearn.metrics import f1_score
    f1_macro = f1_score(
        results["true_label"], results["pred_label"], average="macro"
    )

    metrics = {
        "framework":      "MindsDB",
        "engine":         cfg.get("engine", "huggingface"),
        "model":          cfg.get("model", "distilbert-base-uncased"),
        "accuracy":       accuracy,
        "f1_macro":       f1_macro,
        "n_test":         len(results),
        "inference_time": t.elapsed,
    }

    out_path = os.path.join(results_dir, "metrics", "eval_mindsdb.json")
    save_metrics(metrics, out_path)

    logger.info(f"MindsDB | Accuracy: {accuracy:.4f} | F1: {f1_macro:.4f}")
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MindsDB AG News pipeline")
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     default=47334, type=int)
    parser.add_argument("--config",   default=None)
    args = parser.parse_args()

    DEFAULT_CFG = {
        "model_name":   "ag_news_classifier",
        "engine":       "huggingface",
        "model":        "distilbert-base-uncased",
        "data_cache":   None,
        "results_dir":  "results",
    }
    cfg = DEFAULT_CFG.copy()
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    # 1. Connect
    server = connect_mindsdb(args.host, args.port)

    # 2. Upload data
    train_path, test_path, n_train, n_test = upload_ag_news(server, cfg)

    # 3. Create & train model
    logger.info("=" * 50)
    logger.info("Training MindsDB model")
    logger.info("=" * 50)
    train_time = create_mindsdb_model(server, cfg)

    # 4. Evaluate
    logger.info("=" * 50)
    logger.info("Evaluating MindsDB model")
    logger.info("=" * 50)
    eval_metrics = evaluate_mindsdb(server, cfg, n_test, cfg["results_dir"])

    # 5. Save benchmark
    bench = {
        "framework":        "MindsDB",
        "total_train_time": train_time,
        "n_train":          n_train,
        "n_test":           n_test,
        **eval_metrics,
    }
    save_metrics(bench, os.path.join(cfg["results_dir"], "metrics", "benchmark_mindsdb.json"))

    logger.info("=" * 50)
    logger.info("SUMMARY — MindsDB")
    logger.info("=" * 50)
    logger.info(f"  Engine      : {cfg['engine']}")
    logger.info(f"  Train time  : {train_time:.1f}s")
    logger.info(f"  Accuracy    : {eval_metrics['accuracy']:.4f}")
    logger.info(f"  F1 macro    : {eval_metrics['f1_macro']:.4f}")
    logger.info(
        f"  Key insight : No Python training code — pure SQL interface"
    )


if __name__ == "__main__":
    main()