"""Shared building blocks for the dataset builders.

Both ``tlm_dataset.py`` and ``code_switched_dataset.py`` size their corpora by an
information-equalised token budget and end with the same split/write step. Those
shared pieces live here.
"""

import os
import random

# ---- Byte premium factors (information capacity per token) ----
# Dutch tokens carry ~5% less info -> need more tokens for the same content.
# Chinese tokens carry ~6% more info -> need fewer tokens.
BYTE_PREMIUM_ENGLISH = 1.000000
BYTE_PREMIUM_DUTCH = 1.051606
BYTE_PREMIUM_CHINESE = 0.935966


def adjusted_budget(token_target: int, ratio: float, byte_premium: float) -> int:
    """Per-corpus token budget, equalised for information content.

    Divides the corpus' share of the raw budget by its byte premium so each
    corpus contributes the same amount of *information*, not the same raw token
    count.
    """
    return round((token_target * ratio) / byte_premium)


def sample_to_budget(lines: list[str], token_budget: int) -> list[str]:
    """Keep lines in order until the whitespace-token budget is exhausted."""
    sampled: list[str] = []
    total = 0
    for line in lines:
        n = len(line.split())
        if total + n <= token_budget:
            sampled.append(line)
            total += n
        else:
            break
    return sampled


def _write_lines(path: str, lines: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def write_train_val_split(
    lines: list[str],
    train_path: str,
    val_path: str,
    val_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[list[str], list[str]]:
    """Split ``lines`` into train/validation and write both to disk.

    When ``shuffle`` is True the pool is shuffled before splitting; pass ``seed``
    to reseed first, or leave it ``None`` to preserve an RNG state the caller has
    already established. When ``shuffle`` is False the existing order is kept
    (used when the caller already ordered the lines, e.g. phase 1 then phase 2).
    """
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(lines)

    split_idx = int((1.0 - val_ratio) * len(lines))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    _write_lines(train_path, train_lines)
    _write_lines(val_path, val_lines)
    return train_lines, val_lines
