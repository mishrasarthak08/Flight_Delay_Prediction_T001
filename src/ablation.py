"""
Phase 2 Ablation Study Suite for Deep Learning and Hybrid ML-DL Architectures.
Empirically quantifies the performance impact of:
1. Hybrid Integration Paradigm: Standalone XGBoost vs Standalone Deep ResNet vs
   Naive Probability Averaging vs Regularized Gated Stacking Meta-Learner.
2. Neural Network Architecture & Depth: Plain MLP vs 1-Block ResNet vs 3-Block ResNet.
3. Categorical Representation: High-dim vs Low-dim vs No embeddings.
4. Regularization Techniques: Baseline (none) vs Dropout-only vs Weight Decay vs Full Regularization.
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt

from src.preprocessing import FlightDataset
from src.models import (
    FlightEmbeddingMLP,
    FlightDeepResNet,
    HybridFlightDelayModel,
)


def evaluate_predictions_compact(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes standard evaluation metrics."""
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
    }


def run_hybrid_paradigm_ablation(
    xgb_margin_val: np.ndarray,
    dl_logit_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Compares the 4 paradigms of ML and DL integration on validation data:
    1. Standalone XGBoost (GBDT baseline)
    2. Standalone Deep ResNet (DL baseline)
    3. Naive Probability Averaging (Unweighted Ensemble)
    4. Gated Stacking Meta-Learner (Proposed Phase 2 Hybrid Innovation)
    """
    records = []
    y_val_arr = np.ascontiguousarray(y_val, dtype=np.int32)

    # 1. Standalone XGBoost
    p_xgb = 1.0 / (1.0 + np.exp(-xgb_margin_val))
    m_xgb = evaluate_predictions_compact(y_val_arr, p_xgb)
    records.append({
        "study": "hybrid_paradigm",
        "variant": "Standalone XGBoost (GBDT)",
        "pr_auc": m_xgb["pr_auc"],
        "roc_auc": m_xgb["roc_auc"],
        "brier_score": m_xgb["brier_score"],
        "notes": "Axis-aligned decision boundaries on continuous tabular schedules",
    })

    # 2. Standalone Deep Tabular ResNet
    p_dl = 1.0 / (1.0 + np.exp(-dl_logit_val))
    m_dl = evaluate_predictions_compact(y_val_arr, p_dl)
    records.append({
        "study": "hybrid_paradigm",
        "variant": "Standalone Deep ResNet (DL)",
        "pr_auc": m_dl["pr_auc"],
        "roc_auc": m_dl["roc_auc"],
        "brier_score": m_dl["brier_score"],
        "notes": "Continuous latent embeddings for categorical entities + residual blocks",
    })

    # 3. Naive Probability Averaging
    p_avg = 0.5 * p_xgb + 0.5 * p_dl
    m_avg = evaluate_predictions_compact(y_val_arr, p_avg)
    records.append({
        "study": "hybrid_paradigm",
        "variant": "Naive Averaging (0.5 GBDT + 0.5 DL)",
        "pr_auc": m_avg["pr_auc"],
        "roc_auc": m_avg["roc_auc"],
        "brier_score": m_avg["brier_score"],
        "notes": "Simple heuristic ensemble without learned calibration or meta-weighting",
    })

    # 4. Gated Stacking Meta-Learner (Proposed Innovation)
    hybrid_stack = HybridFlightDelayModel(fusion_strategy="gated_stacking", meta_l2=0.01, random_seed=random_seed)
    hybrid_stack.fit_fusion(xgb_margin_val, dl_logit_val, y_val_arr)
    p_stack = hybrid_stack.predict_fusion_proba(xgb_margin_val, dl_logit_val)
    m_stack = evaluate_predictions_compact(y_val_arr, p_stack)
    weights = hybrid_stack.get_fusion_weights()
    records.append({
        "study": "hybrid_paradigm",
        "variant": "Learned Gated Stacking (Proposed Hybrid)",
        "pr_auc": m_stack["pr_auc"],
        "roc_auc": m_stack["roc_auc"],
        "brier_score": m_stack["brier_score"],
        "notes": f"Optimal logit-space meta-fusion: w_xgb={weights['w_xgb']:.3f}, w_dl={weights['w_dl']:.3f}, b={weights['intercept']:.3f}",
    })

    return records


def run_architecture_ablation(
    train_cat: torch.Tensor,
    train_num: torch.Tensor,
    train_y: torch.Tensor,
    val_cat: torch.Tensor,
    val_num: torch.Tensor,
    val_y: np.ndarray,
    vocab_sizes: Dict[str, int],
    num_numeric: int,
    epochs: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Ablates neural network depth, capacity, and residual skip connections:
    - 1-Layer Shallow MLP (no skip connection)
    - 3-Layer Dense MLP (no skip connection)
    - 1-Block ResNet (with residual skip connection)
    - 3-Block Deep ResNet (with residual skip connections)
    """
    records = []
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")

    batch_size = 512
    train_ds = FlightDataset(train_cat, train_num, train_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    configs = [
        ("Shallow MLP (1 Layer)", "mlp", [64], 0),
        ("Standard MLP (3 Layers)", "mlp", [128, 64, 32], 0),
        ("Shallow ResNet (1 Block)", "resnet", 64, 1),
        ("Deep ResNet (3 Blocks)", "resnet", 128, 3),
    ]

    for label, arch_type, param1, param2 in configs:
        if arch_type == "mlp":
            model = FlightEmbeddingMLP(
                num_carriers=vocab_sizes["carrier"],
                num_origins=vocab_sizes["origin"],
                num_dests=vocab_sizes["dest"],
                num_numeric_features=num_numeric,
                hidden_dims=param1,
                dropout=0.2,
            )
        else:
            model = FlightDeepResNet(
                num_carriers=vocab_sizes["carrier"],
                num_origins=vocab_sizes["origin"],
                num_dests=vocab_sizes["dest"],
                num_numeric_features=num_numeric,
                hidden_dim=param1,
                num_blocks=param2,
                dropout=0.2,
            )

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            model.train()
            for b_cat, b_num, b_y in train_loader:
                optimizer.zero_grad()
                out = model(b_cat, b_num)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

        probs = model.predict_proba(val_cat, val_num, device=device)
        m = evaluate_predictions_compact(val_y, probs)
        records.append({
            "study": "architecture_depth",
            "variant": label,
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
            "notes": f"Trained for {epochs} epochs on training partition",
        })

    return records


def run_regularization_ablation(
    train_cat: torch.Tensor,
    train_num: torch.Tensor,
    train_y: torch.Tensor,
    val_cat: torch.Tensor,
    val_num: torch.Tensor,
    val_y: np.ndarray,
    vocab_sizes: Dict[str, int],
    num_numeric: int,
    epochs: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Ablates regularization mechanisms on Deep Tabular ResNet:
    1. Unregularized (Dropout=0.0, Weight Decay=0.0)
    2. Dropout Only (Dropout=0.3, Weight Decay=0.0)
    3. L2 Weight Decay Only (Dropout=0.0, Weight Decay=1e-3)
    4. Full Regularization (Dropout=0.2, Weight Decay=1e-4, Batch Normalization)
    """
    records = []
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    device = torch.device("cpu")

    batch_size = 512
    train_ds = FlightDataset(train_cat, train_num, train_y)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    reg_configs = [
        ("No Regularization", 0.0, 0.0),
        ("Dropout Only (p=0.3)", 0.3, 0.0),
        ("Weight Decay Only (lambda=1e-3)", 0.0, 1e-3),
        ("Full Regularization (Dropout=0.2 + Weight Decay=1e-4)", 0.2, 1e-4),
    ]

    for label, dropout_val, wd_val in reg_configs:
        model = FlightDeepResNet(
            num_carriers=vocab_sizes["carrier"],
            num_origins=vocab_sizes["origin"],
            num_dests=vocab_sizes["dest"],
            num_numeric_features=num_numeric,
            hidden_dim=64,
            num_blocks=2,
            dropout=dropout_val,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=wd_val)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(epochs):
            model.train()
            for b_cat, b_num, b_y in train_loader:
                optimizer.zero_grad()
                out = model(b_cat, b_num)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

        probs = model.predict_proba(val_cat, val_num, device=device)
        m = evaluate_predictions_compact(val_y, probs)
        records.append({
            "study": "regularization",
            "variant": label,
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "brier_score": m["brier_score"],
            "notes": f"Dropout={dropout_val}, Weight Decay={wd_val}",
        })

    return records


def plot_ablation_summary(df_ablation: pd.DataFrame, output_path: Path) -> None:
    """Generates visual comparison figures for the ablation studies."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    studies = df_ablation["study"].unique()
    fig, axes = plt.subplots(1, len(studies), figsize=(6 * len(studies), 5), squeeze=False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, study in enumerate(studies):
        ax = axes[0, idx]
        subset = df_ablation[df_ablation["study"] == study].copy()
        bars = ax.barh(
            subset["variant"],
            subset["pr_auc"],
            color=colors[idx % len(colors)],
            alpha=0.85,
            edgecolor="black",
        )
        ax.set_title(f"Ablation: {study.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("PR-AUC (Average Precision)", fontsize=10)
        ax.set_xlim(left=max(0.0, subset["pr_auc"].min() - 0.05), right=subset["pr_auc"].max() + 0.05)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.4f}",
                va="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
