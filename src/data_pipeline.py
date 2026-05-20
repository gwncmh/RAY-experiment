"""
data_pipeline.py — Load AG News và ETL bằng Ray Data

Pipeline:
    HuggingFace datasets  →  Ray Dataset  →  tokenize (map_batches)  →  train/test split

Dùng:
    from src.data_pipeline import build_datasets
    train_ds, test_ds = build_datasets(cfg)
"""

from __future__ import annotations

import ray
import ray.data
from datasets import load_dataset
from transformers import AutoTokenizer

from src.utils import Timer, get_logger, calc_throughput

logger = get_logger(__name__)


# ── Tokenizer (singleton, tránh load lại nhiều lần) ───────────────────────────

_tokenizer: AutoTokenizer | None = None


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None or _tokenizer.name_or_path != model_name:
        logger.info(f"Loading tokenizer: {model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ── Tokenize function (chạy distributed bởi Ray Data) ─────────────────────────

def _make_tokenize_fn(model_name: str, max_length: int):
    """
    Trả về function tokenize để truyền vào map_batches.
    Đóng gói model_name và max_length vào closure.
    """
    def tokenize_batch(batch: dict) -> dict:
        tok = _get_tokenizer(model_name)
        # FIX: convert sang list[str] vì Ray Data 2.31+ truyền numpy array
        texts = batch["text"].tolist() if hasattr(batch["text"], "tolist") else list(batch["text"])
        encoded = tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        return {
            "input_ids":      encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label":          batch["label"],
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def load_ray_datasets(
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    cache_dir: str | None = None,
) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Load AG News từ HuggingFace, chuyển sang Ray Dataset, rồi tokenize.

    Args:
        model_name: tên tokenizer (phải khớp với model dùng trong train.py).
        max_length:  độ dài padding/truncation.
        cache_dir:   thư mục cache HuggingFace (None = mặc định ~/.cache).

    Returns:
        (train_ds, test_ds) — đã tokenize, sẵn sàng để train/evaluate.
    """
    # 1. Load raw dataset
    logger.info("Downloading AG News from HuggingFace...")
    with Timer("HuggingFace download") as t:
        hf_ds = load_dataset("ag_news", cache_dir=cache_dir)

    logger.info(
        f"AG News loaded — train: {len(hf_ds['train'])}, "
        f"test: {len(hf_ds['test'])} samples"
    )

    # 2. Chuyển sang Ray Dataset
    logger.info("Converting to Ray Dataset...")
    train_ray = ray.data.from_huggingface(hf_ds["train"])
    test_ray  = ray.data.from_huggingface(hf_ds["test"])

    # 3. Tokenize bằng map_batches (Ray tự song song hoá)
    tokenize_fn = _make_tokenize_fn(model_name, max_length)

    logger.info(f"Tokenizing with max_length={max_length}...")
    with Timer("Tokenization (train)") as tt:
        train_tok = train_ray.map_batches(tokenize_fn, batch_size=256)
        # Trigger materialize để đo thời gian thật
        n_train = train_tok.count()

    throughput = calc_throughput(n_train, tt.elapsed)
    logger.info(
        f"Tokenized {n_train} train samples | "
        f"throughput: {throughput:.0f} samples/sec"
    )

    test_tok = test_ray.map_batches(tokenize_fn, batch_size=256)

    return train_tok, test_tok


def build_datasets(cfg: dict) -> tuple[ray.data.Dataset, ray.data.Dataset]:
    """
    Wrapper đọc config dict rồi gọi load_ray_datasets.
    Dùng trong experiments/run_ray.py.

    cfg keys cần có:
        model        (str)
        max_length   (int)
        data_cache   (str | None, optional)
    """
    return load_ray_datasets(
        model_name=cfg.get("model", "bert-base-uncased"),
        max_length=cfg.get("max_length", 128),
        cache_dir=cfg.get("data_cache", None),
    )
