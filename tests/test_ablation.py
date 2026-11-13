"""Unit tests for Phase 2 ablation study module."""

import numpy as np
import pandas as pd
import torch
import pytest

from src.ablation import (
    evaluate_predictions_compact,
    run_hybrid_paradigm_ablation,
    run_architecture_ablation,
    run_regularization_ablation,
    plot_ablation_summary,
)


def test_evaluate_predictions_compact():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    metrics = evaluate_predictions_compact(y_true, y_prob)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert metrics["roc_auc"] > 0.8


def test_run_hybrid_paradigm_ablation():
    np.random.seed(42)
    n = 60
    xgb_m = np.random.randn(n).astype(np.float32)
    dl_l = np.random.randn(n).astype(np.float32)
    y = np.random.randint(0, 2, n).astype(np.int32)
    records = run_hybrid_paradigm_ablation(xgb_m, dl_l, y, random_seed=42)
    assert len(records) == 4
    variants = [r["variant"] for r in records]
    assert "Standalone XGBoost (GBDT)" in variants
    assert "Standalone Deep ResNet (DL)" in variants
    assert "Learned Gated Stacking (Proposed Hybrid)" in variants


def test_plot_ablation_summary(tmp_path):
    records = [
        {"study": "hybrid_paradigm", "variant": "XGBoost", "pr_auc": 0.35, "roc_auc": 0.65, "brier_score": 0.18},
        {"study": "hybrid_paradigm", "variant": "Hybrid", "pr_auc": 0.40, "roc_auc": 0.69, "brier_score": 0.16},
    ]
    df = pd.DataFrame(records)
    out_file = tmp_path / "ablation_test.png"
    plot_ablation_summary(df, out_file)
    assert out_file.exists()
    assert out_file.stat().st_size > 0
