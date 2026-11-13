"""Unit tests for Phase 2 Hybrid ML-DL architecture and integration."""

import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path

from src.models import (
    XGBoostModel,
    FlightEmbeddingMLP,
    FlightDeepResNet,
    HybridFlightDelayModel,
)
from src.config import ModelHyperparams, ALL_MODEL_FEATURES, TARGET_COL
from src.preprocessing import FlightTabularPreprocessor, PyTorchPreprocessor
from src.inference import FlightDelayPredictor


def test_hybrid_model_fusion_mechanics():
    np.random.seed(42)
    n = 100
    xgb_m = np.random.randn(n).astype(np.float32)
    dl_l = np.random.randn(n).astype(np.float32)
    y_val = (xgb_m + dl_l > 0).astype(np.int32)

    # Test gated stacking
    hybrid = HybridFlightDelayModel(fusion_strategy="gated_stacking", meta_l2=0.01)
    hybrid.fit_fusion(xgb_m, dl_l, y_val)
    assert hybrid.is_fitted
    weights = hybrid.get_fusion_weights()
    assert "w_xgb" in weights
    assert "w_dl" in weights
    assert "intercept" in weights

    # Probabilities must be strictly bounded in [0, 1]
    probs = hybrid.predict_fusion_proba(xgb_m[:10], dl_l[:10])
    assert len(probs) == 10
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Predictions must be binary
    preds = (probs >= 0.5).astype(int)
    assert set(np.unique(preds)).issubset({0, 1})


def test_hybrid_weighted_average_strategy():
    np.random.seed(42)
    n = 50
    xgb_m = np.random.randn(n).astype(np.float32)
    dl_l = np.random.randn(n).astype(np.float32)
    y_val = (xgb_m > 0).astype(np.int32)

    hybrid = HybridFlightDelayModel(fusion_strategy="weighted_average")
    hybrid.fit_fusion(xgb_m, dl_l, y_val)
    weights = hybrid.get_fusion_weights()
    assert pytest.approx(weights["w_xgb"] + weights["w_dl"]) == 1.0
    assert weights["w_xgb"] >= 0.0
    assert weights["w_dl"] >= 0.0


def test_hybrid_serialization(tmp_path):
    hybrid = HybridFlightDelayModel(fusion_strategy="gated_stacking")
    xgb_m = np.array([0.5, -0.5, 1.0], dtype=np.float32)
    dl_l = np.array([0.2, -0.1, 0.8], dtype=np.float32)
    y_val = np.array([1, 0, 1], dtype=np.int32)
    hybrid.fit_fusion(xgb_m, dl_l, y_val)

    save_path = tmp_path / "test_hybrid.pkl"
    hybrid.save(save_path)
    assert save_path.exists()

    loaded = HybridFlightDelayModel.load(save_path)
    assert loaded.is_fitted
    p1 = hybrid.predict_fusion_proba(xgb_m, dl_l)
    p2 = loaded.predict_fusion_proba(xgb_m, dl_l)
    np.testing.assert_allclose(p1, p2)


def test_hybrid_predictor_e2e_inference():
    predictor = FlightDelayPredictor("hybrid_model")
    assert predictor.is_hybrid
    assert predictor.model_name == "hybrid_model"

    demo = pd.read_csv(Path(__file__).resolve().parents[1] / "results/reports/demo_flights.csv").iloc[0]
    flight = {
        "flight_date": str(demo["flight_date"]),
        "carrier": str(demo["carrier"]),
        "origin": str(demo["origin"]),
        "dest": str(demo["dest"]),
        "sched_dep_time": int(demo["sched_dep_time"]),
        "sched_arr_time": int(demo["sched_arr_time"]),
        "distance": float(demo["distance"]),
    }
    result = predictor.predict_single(flight)
    assert "predicted_delay_prob_raw" in result
    assert "predicted_delay_prob_calibrated" in result
    assert 0.0 <= result["predicted_delay_prob_calibrated"] <= 1.0
    assert result["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
