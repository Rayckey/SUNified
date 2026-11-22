"""Stage metrics with explicit support counts and separate evaluation settings."""

import numpy as np


def stage_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(axis=1)
    correct = confusion.diagonal()
    per_stage = [float(hit / count) if count else None for hit, count in zip(correct, support)]
    represented = [value for value in per_stage if value is not None]
    return {"confusion": confusion.tolist(), "support": support.tolist(), "per_stage_accuracy": per_stage,
            "macro_accuracy": float(np.mean(represented)) if represented else None,
            "accuracy": float(correct.sum() / support.sum()) if support.sum() else None,
            "macro_definition": "mean over stages with nonzero support"}


def add_confusion(confusion, labels, predictions):
    labels, predictions = np.asarray(labels).reshape(-1), np.asarray(predictions).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("Stage labels and predictions have different shapes")
    valid = labels != -1
    if np.any((labels[valid] < 0) | (labels[valid] >= len(confusion))) or np.any(
        (predictions[valid] < 0) | (predictions[valid] >= len(confusion))
    ):
        raise ValueError("Stage metric inputs are outside the declared vocabulary")
    np.add.at(confusion, (labels[valid], predictions[valid]), 1)
