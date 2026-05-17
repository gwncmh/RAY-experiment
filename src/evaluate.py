"""
evaluate.py — Đánh giá model trên AG News test set

Dùng:
    from src.evaluate import evaluate_model
    metrics = evaluate_model(model, test_ds, cfg)
"""

from __future__ import annotations

import os
import torch
import numpy as np
from typing import Any

import ray.data
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils import get_logger, save_metrics, Timer, LABEL_NAMES

logger = get_logger(__name__)


# ── Core evaluation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def _predict(model: torch.nn.Module, test_ds: ray.data.Dataset, batch_size: int = 64):
    """
    Chạy inference trên toàn bộ test set, trả về (y_true, y_pred).
    """
    device = next(model.parameters()).device
    model.eval()

    all_preds, all_labels = [], []

    for batch in test_ds.iter_torch_batches(batch_size=batch_size, dtypes=torch.long):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].cpu().numpy()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    test_ds: ray.data.Dataset,
    cfg: dict,
    results_dir: str = "results",
) -> dict[str, Any]:
    """
    Tính accuracy, macro F1, per-class F1 và confusion matrix.

    Args:
        model:       model đã train (PyTorch).
        test_ds:     Ray Dataset test đã tokenize.
        cfg:         config dict.
        results_dir: thư mục lưu kết quả.

    Returns:
        dict metrics gồm accuracy, f1_macro, f1_per_class.
    """
    batch_size = cfg.get("eval_batch_size", 64)

    logger.info(f"Evaluating on test set (batch_size={batch_size})...")

    with Timer("Evaluation") as t:
        y_true, y_pred = _predict(model, test_ds, batch_size)

    accuracy  = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, average="macro")
    f1_each   = f1_score(y_true, y_pred, average=None).tolist()
    report    = classification_report(
        y_true, y_pred,
        target_names=list(LABEL_NAMES.values()),
        digits=4,
    )
    conf_mat  = confusion_matrix(y_true, y_pred).tolist()

    logger.info(f"\n{report}")

    metrics = {
        "framework":    cfg.get("framework", "ray"),
        "model":        cfg.get("model", "bert-base-uncased"),
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_per_class": {
            LABEL_NAMES[i]: round(f1_each[i], 4)
            for i in range(len(f1_each))
        },
        "confusion_matrix": conf_mat,
        "eval_time_sec": t.elapsed,
        "n_test_samples": len(y_true),
    }

    out_path = os.path.join(results_dir, "metrics", "eval_metrics.json")
    save_metrics(metrics, out_path)

    logger.info(
        f"Accuracy: {accuracy:.4f} | F1 macro: {f1_macro:.4f} | "
        f"Eval time: {t.elapsed:.2f}s"
    )

    return metrics