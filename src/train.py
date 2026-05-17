"""
train.py — Distributed fine-tuning với Ray Train

Pipeline:
    Ray Dataset  →  TorchTrainer (N workers)  →  checkpoint  →  metrics JSON

Dùng:
    from src.train import run_training
    result = run_training(train_ds, cfg)
"""

from __future__ import annotations

import os
import time
import torch
from typing import Any

import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from src.utils import get_logger, save_metrics, calc_throughput

logger = get_logger(__name__)


# ── Training function (chạy trên mỗi worker) ──────────────────────────────────

def _train_loop(config: dict) -> None:
    """
    Hàm này chạy trên TỪNG Ray worker — không import logger ngoài ở đây.
    Ray tự chia shard dataset cho mỗi worker.
    """
    from transformers import AutoModelForSequenceClassification

    # ── Model ──
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"],
        num_labels=config["num_labels"],
    )
    model = train.torch.prepare_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Warmup + linear decay scheduler
    from transformers import get_linear_schedule_with_warmup
    total_steps = config.get("total_steps", 1000)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── Dataset shard ──
    ds = train.get_dataset_shard("train")

    for epoch in range(config["epochs"]):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()

        for batch in ds.iter_torch_batches(
            batch_size=config["batch_size"],
            dtypes=torch.long,
        ):
            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels         = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            optimizer.zero_grad()
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            preds    = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += outputs.loss.item() * labels.size(0)

        elapsed    = time.perf_counter() - t0
        throughput = calc_throughput(total, elapsed)

        metrics = {
            "epoch":       epoch + 1,
            "train_loss":  total_loss / max(total, 1),
            "train_acc":   correct / max(total, 1),
            "throughput":  throughput,
            "epoch_time":  elapsed,
        }

        # Lưu model checkpoint để load_checkpoint() dùng được
        import os
        ckpt_dir = train.get_context().get_local_rank()  # 0 = main worker
        if ckpt_dir == 0:
            os.makedirs("checkpoint_tmp", exist_ok=True)
            torch.save(model.module.state_dict()
                       if hasattr(model, "module") else model.state_dict(),
                       "checkpoint_tmp/model.pt")
            train.report(metrics, checkpoint=train.Checkpoint.from_directory("checkpoint_tmp"))
        else:
            train.report(metrics)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_training(
    train_ds: ray.data.Dataset,
    cfg: dict,
    results_dir: str = "results",
) -> dict[str, Any]:
    """
    Khởi động TorchTrainer với N workers theo config.

    Args:
        train_ds:    Ray Dataset đã tokenize (từ data_pipeline.py).
        cfg:         config dict — xem configs/ray_config.yaml.
        results_dir: thư mục lưu metrics JSON.

    Returns:
        dict chứa metrics của epoch cuối.
    """
    num_workers  = cfg.get("num_workers", 2)
    use_gpu      = cfg.get("use_gpu", False)

    logger.info(
        f"Starting training — "
        f"workers={num_workers}, gpu={use_gpu}, "
        f"epochs={cfg.get('epochs', 3)}, batch={cfg.get('batch_size', 32)}"
    )

    # Tính total_steps để scheduler dùng
    approx_samples   = 120_000  # AG News train size
    steps_per_epoch  = approx_samples // (cfg.get("batch_size", 32) * num_workers)
    total_steps      = steps_per_epoch * cfg.get("epochs", 3)

    train_loop_cfg = {
        "model":       cfg.get("model", "bert-base-uncased"),
        "num_labels":  cfg.get("num_labels", 4),
        "lr":          cfg.get("lr", 2e-5),
        "weight_decay": cfg.get("weight_decay", 0.01),
        "epochs":      cfg.get("epochs", 3),
        "batch_size":  cfg.get("batch_size", 32),
        "total_steps": total_steps,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=_train_loop,
        train_loop_config=train_loop_cfg,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
        ),
        run_config=RunConfig(
            name="ray_ag_news",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,  # chỉ giữ checkpoint tốt nhất
            ),
        ),
        datasets={"train": train_ds},
    )

    t_start = time.perf_counter()
    result  = trainer.fit()
    total_time = time.perf_counter() - t_start

    final_metrics = result.metrics or {}
    final_metrics["total_train_time"] = total_time
    final_metrics["num_workers"]      = num_workers
    final_metrics["framework"]        = "ray"

    # Lưu metrics ra disk
    out_path = os.path.join(results_dir, "metrics", "ray_train_metrics.json")
    save_metrics(final_metrics, out_path)

    logger.info(
        f"Training done in {total_time:.1f}s | "
        f"final acc: {final_metrics.get('train_acc', 0):.4f}"
    )

    return final_metrics