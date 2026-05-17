"""
model.py — Định nghĩa BERT classifier cho AG News (4 lớp)

Dùng:
    from src.model import build_model
    model = build_model("bert-base-uncased", num_labels=4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

from src.utils import get_logger

logger = get_logger(__name__)


# ── Model builder ──────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "bert-base-uncased",
    num_labels: int = 4,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Load pretrained BERT và gắn classification head.

    Args:
        model_name: checkpoint HuggingFace (bert-base-uncased, distilbert, v.v.)
        num_labels: số lớp phân loại — AG News = 4.
        dropout:    dropout rate trên classification head.

    Returns:
        PyTorch model sẵn sàng để train.
    """
    logger.info(f"Loading model: {model_name} (num_labels={num_labels})")

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,  # an toàn khi load checkpoint cũ
    )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded — {n_params / 1e6:.1f}M parameters")

    return model


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str) -> None:
    """Lưu state_dict ra disk."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """Load state_dict vào model đã khởi tạo."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    logger.info(f"Checkpoint loaded ← {path}")
    return model