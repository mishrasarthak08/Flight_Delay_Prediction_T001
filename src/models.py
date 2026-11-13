"""
Model architectures for Flight Delay Prediction.
Includes:
- Model A: Scikit-learn Logistic Regression baseline
- Model B: XGBoost Gradient Boosted Decision Trees
- Model C: PyTorch Embedding MLP with learned entity embeddings for carrier, origin, and destination
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.config import ModelHyperparams


class LogisticRegressionBaseline:
    """Logistic Regression baseline classifier wrapper."""

    def __init__(
        self, hyperparams: ModelHyperparams = ModelHyperparams(), random_seed: int = 42
    ):
        self.model = LogisticRegression(
            C=hyperparams.lr_c,
            max_iter=hyperparams.lr_max_iter,
            solver=hyperparams.lr_solver,
            class_weight=hyperparams.lr_class_weight,
            random_state=random_seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class XGBoostModel:
    """XGBoost Gradient Boosted Classifier wrapper."""

    def __init__(
        self, hyperparams: ModelHyperparams = ModelHyperparams(), random_seed: int = 42
    ):
        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.model = xgb.XGBClassifier(
            n_estimators=hyperparams.xgb_n_estimators,
            max_depth=hyperparams.xgb_max_depth,
            learning_rate=hyperparams.xgb_learning_rate,
            subsample=hyperparams.xgb_subsample,
            colsample_bytree=hyperparams.xgb_colsample_bytree,
            eval_metric=hyperparams.xgb_eval_metric,
            tree_method=hyperparams.xgb_tree_method,
            random_state=random_seed,
            n_jobs=hyperparams.xgb_n_jobs,
            early_stopping_rounds=hyperparams.xgb_early_stopping_rounds,
            reg_lambda=hyperparams.xgb_reg_lambda,
            min_child_weight=hyperparams.xgb_min_child_weight,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostModel":
        X_train_c = np.ascontiguousarray(X_train, dtype=np.float32)
        y_train_c = np.ascontiguousarray(y_train, dtype=np.float32)

        if X_val is not None and y_val is not None:
            X_val_c = np.ascontiguousarray(X_val, dtype=np.float32)
            y_val_c = np.ascontiguousarray(y_val, dtype=np.float32)
            eval_set = [(X_val_c, y_val_c)]
            self.model.set_params(early_stopping_rounds=self.hyperparams.xgb_early_stopping_rounds)
        else:
            eval_set = None
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(
            X_train_c,
            y_train_c,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.ascontiguousarray(X, dtype=np.float32))[:, 1]

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Computes raw tree margins (log-odds) before logistic transformation."""
        return self.model.predict(
            np.ascontiguousarray(X, dtype=np.float32), output_margin=True
        )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importances(self, feature_names: List[str]) -> Dict[str, float]:
        importances = self.model.feature_importances_
        return {name: float(imp) for name, imp in zip(feature_names, importances)}


class FlightEmbeddingMLP(nn.Module):
    """
    Compact PyTorch Neural Network using learned categorical embeddings
    for airline carrier, origin airport, and destination airport,
    concatenated with standardized numeric temporal and operational features.
    """

    def __init__(
        self,
        num_carriers: int,
        num_origins: int,
        num_dests: int,
        num_numeric_features: int,
        carrier_emb_dim: int = 16,
        origin_emb_dim: int = 16,
        dest_emb_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_carriers = num_carriers
        self.num_origins = num_origins
        self.num_dests = num_dests
        self.num_numeric_features = num_numeric_features
        self.carrier_emb_dim = carrier_emb_dim
        self.origin_emb_dim = origin_emb_dim
        self.dest_emb_dim = dest_emb_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout_rate = dropout

        # Categorical Embeddings
        self.carrier_embedding = nn.Embedding(
            num_carriers, carrier_emb_dim, padding_idx=0
        )
        self.origin_embedding = nn.Embedding(num_origins, origin_emb_dim, padding_idx=0)
        self.dest_embedding = nn.Embedding(num_dests, dest_emb_dim, padding_idx=0)

        # Total input dimension into MLP
        total_in_dim = (
            carrier_emb_dim + origin_emb_dim + dest_emb_dim + num_numeric_features
        )

        # Dense MLP Backbone
        layers: List[nn.Module] = []
        curr_dim = total_in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        # Binary classification output logit
        layers.append(nn.Linear(curr_dim, 1))
        self.backbone = nn.Sequential(*layers)

    def forward(
        self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        cat_tensor: (batch_size, 3) where columns are [carrier_idx, origin_idx, dest_idx]
        num_tensor: (batch_size, num_numeric_features)
        Returns: logits (batch_size, 1)
        """
        c_emb = self.carrier_embedding(cat_tensor[:, 0])
        o_emb = self.origin_embedding(cat_tensor[:, 1])
        d_emb = self.dest_embedding(cat_tensor[:, 2])

        # Concatenate embeddings and numeric features
        x = torch.cat([c_emb, o_emb, d_emb, num_tensor], dim=1)
        logits = self.backbone(x)
        return logits

    def predict_proba(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """Computes sigmoid probabilities in evaluation mode."""
        self.eval()
        with torch.no_grad():
            cat_dev = cat_tensor.to(device)
            num_dev = num_tensor.to(device)
            logits = self(cat_dev, num_dev)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict_logits(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """Computes raw model logits in evaluation mode."""
        self.eval()
        with torch.no_grad():
            cat_dev = cat_tensor.to(device)
            num_dev = num_tensor.to(device)
            logits = self(cat_dev, num_dev)
            return logits.cpu().numpy().flatten()

    def save_checkpoint(
        self, filepath: Path, config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        """Saves model weights and architecture configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "architecture": {
                "num_carriers": self.num_carriers,
                "num_origins": self.num_origins,
                "num_dests": self.num_dests,
                "num_numeric_features": self.num_numeric_features,
                "carrier_emb_dim": self.carrier_emb_dim,
                "origin_emb_dim": self.origin_emb_dim,
                "dest_emb_dim": self.dest_emb_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout_rate,
            },
            "config": config_dict or {},
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls, filepath: Path, device: torch.device = torch.device("cpu")
    ) -> "FlightEmbeddingMLP":
        """Loads model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        arch = checkpoint["architecture"]
        model = cls(**arch)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


class TabularResidualBlock(nn.Module):
    """
    Residual block with normalization, activation, and dropout.
    Preserves gradient flow across deep tabular architectures:
    h_{l+1} = h_l + Dropout(Linear(ReLU(Norm(Linear(Norm(h_l))))))
    """

    def __init__(self, dim: int, dropout: float = 0.2, use_layernorm: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim) if use_layernorm else nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim) if use_layernorm else nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.norm1(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout1(out)
        out = self.norm2(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return res + out


class FlightDeepResNet(nn.Module):
    """
    Deep Tabular ResNet for Flight Delay Prediction (Phase 2 Deep Learning Architecture).
    Projects high-cardinality discrete categories into dense entity embeddings,
    projects numeric inputs into a shared dimension, and passes through stacked
    residual blocks with skip connections to prevent gradient degradation.
    """

    def __init__(
        self,
        num_carriers: int,
        num_origins: int,
        num_dests: int,
        num_numeric_features: int,
        carrier_emb_dim: int = 16,
        origin_emb_dim: int = 16,
        dest_emb_dim: int = 32,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.2,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.num_carriers = num_carriers
        self.num_origins = num_origins
        self.num_dests = num_dests
        self.num_numeric_features = num_numeric_features
        self.carrier_emb_dim = carrier_emb_dim
        self.origin_emb_dim = origin_emb_dim
        self.dest_emb_dim = dest_emb_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout
        self.use_layernorm = use_layernorm

        # Entity Embeddings
        self.carrier_embedding = nn.Embedding(
            num_carriers, carrier_emb_dim, padding_idx=0
        )
        self.origin_embedding = nn.Embedding(num_origins, origin_emb_dim, padding_idx=0)
        self.dest_embedding = nn.Embedding(num_dests, dest_emb_dim, padding_idx=0)

        # Total input dimension before projection
        raw_dim = carrier_emb_dim + origin_emb_dim + dest_emb_dim + num_numeric_features

        # Projection into hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stacked Residual Blocks
        self.res_blocks = nn.ModuleList(
            [
                TabularResidualBlock(
                    hidden_dim, dropout=dropout, use_layernorm=use_layernorm
                )
                for _ in range(num_blocks)
            ]
        )

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor
    ) -> torch.Tensor:
        c_emb = self.carrier_embedding(cat_tensor[:, 0])
        o_emb = self.origin_embedding(cat_tensor[:, 1])
        d_emb = self.dest_embedding(cat_tensor[:, 2])

        x = torch.cat([c_emb, o_emb, d_emb, num_tensor], dim=1)
        h = self.input_projection(x)
        for block in self.res_blocks:
            h = block(h)
        logits = self.head(h)
        return logits

    def predict_proba(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            cat_dev = cat_tensor.to(device)
            num_dev = num_tensor.to(device)
            logits = self(cat_dev, num_dev)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict_logits(
        self,
        cat_tensor: torch.Tensor,
        num_tensor: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            cat_dev = cat_tensor.to(device)
            num_dev = num_tensor.to(device)
            logits = self(cat_dev, num_dev)
            return logits.cpu().numpy().flatten()

    def save_checkpoint(
        self, filepath: Path, config_dict: Optional[Dict[str, Any]] = None
    ) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "architecture": {
                "num_carriers": self.num_carriers,
                "num_origins": self.num_origins,
                "num_dests": self.num_dests,
                "num_numeric_features": self.num_numeric_features,
                "carrier_emb_dim": self.carrier_emb_dim,
                "origin_emb_dim": self.origin_emb_dim,
                "dest_emb_dim": self.dest_emb_dim,
                "hidden_dim": self.hidden_dim,
                "num_blocks": self.num_blocks,
                "dropout": self.dropout_rate,
                "use_layernorm": self.use_layernorm,
            },
            "config": config_dict or {},
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls, filepath: Path, device: torch.device = torch.device("cpu")
    ) -> "FlightDeepResNet":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        arch = checkpoint["architecture"]
        model = cls(**arch)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model


class HybridFlightDelayModel:
    """
    Phase 2 Hybrid ML-DL Architecture for Flight Delay Prediction.
    
    Technical Rationale:
    1. GBDT (XGBoost) forms orthogonal, axis-aligned decision trees that excel at
       discovering sharp thresholds on continuous operational features (e.g. airport
       hourly departure spikes, calendar trends, flight route distance).
    2. Deep Tabular Networks (PyTorch ResNet / Embedding MLP) learn smooth continuous
       latent geometric manifolds for categorical entities (carrier, origin, destination),
       capturing semantic similarities and generalizing across rare airline routes.
    3. The Logit-Space Gated Stacking Meta-Learner unifies both predictors:
       z = w_xgb * margin_xgb + w_dl * logit_dl + b_0
       Fit on held-out validation labels to prevent leakage, guaranteeing optimal
       error-variance reduction over any single constituent family.
    """

    def __init__(
        self,
        xgb_model: Optional[XGBoostModel] = None,
        dl_model: Optional[nn.Module] = None,
        fusion_strategy: str = "gated_stacking",
        meta_l2: float = 0.01,
        random_seed: int = 42,
    ):
        self.xgb_model = xgb_model
        self.dl_model = dl_model
        self.fusion_strategy = fusion_strategy
        self.meta_l2 = meta_l2
        self.random_seed = random_seed
        self.is_fitted = False
        # Meta-learner parameters: z = w_xgb * m_xgb + w_dl * l_dl + b
        self.w_xgb: float = 0.5
        self.w_dl: float = 0.5
        self.intercept: float = 0.0
        self.meta_lr = None

    def fit_fusion(
        self,
        xgb_margin_val: np.ndarray,
        dl_logit_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "HybridFlightDelayModel":
        """
        Fits the regularized stacking meta-learner on held-out validation margins/logits.
        Uses L2-regularized logistic regression over the two raw logit streams.
        """
        X_meta = np.column_stack([xgb_margin_val, dl_logit_val]).astype(np.float32)
        y_val_arr = np.ascontiguousarray(y_val, dtype=np.int32)

        if self.fusion_strategy == "weighted_average":
            # Simple grid search for convex combination parameter alpha
            best_alpha = 0.5
            best_loss = float("inf")
            from sklearn.metrics import brier_score_loss

            p_xgb = 1.0 / (1.0 + np.exp(-xgb_margin_val))
            p_dl = 1.0 / (1.0 + np.exp(-dl_logit_val))
            for alpha in np.linspace(0.0, 1.0, 101):
                p_comb = alpha * p_xgb + (1 - alpha) * p_dl
                loss = brier_score_loss(y_val_arr, p_comb)
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = alpha
            self.w_xgb = float(best_alpha)
            self.w_dl = float(1.0 - best_alpha)
            self.intercept = 0.0
        else:
            # Gated stacking: logistic regression on log-odds space
            # C is inverse of regularization strength
            c_val = 1.0 / max(self.meta_l2, 1e-5)
            self.meta_lr = LogisticRegression(
                C=c_val,
                fit_intercept=True,
                solver="lbfgs",
                random_state=self.random_seed,
            )
            self.meta_lr.fit(X_meta, y_val_arr)
            self.w_xgb = float(self.meta_lr.coef_[0, 0])
            self.w_dl = float(self.meta_lr.coef_[0, 1])
            self.intercept = float(self.meta_lr.intercept_[0])

        self.is_fitted = True
        return self

    def predict_fusion_proba(
        self, xgb_margin: np.ndarray, dl_logit: np.ndarray
    ) -> np.ndarray:
        """
        Combines raw margins and logits using fitted meta-parameters.
        """
        if self.fusion_strategy == "weighted_average":
            p_xgb = 1.0 / (1.0 + np.exp(-xgb_margin))
            p_dl = 1.0 / (1.0 + np.exp(-dl_logit))
            return self.w_xgb * p_xgb + self.w_dl * p_dl
        elif self.meta_lr is not None:
            X_meta = np.column_stack([xgb_margin, dl_logit]).astype(np.float32)
            return self.meta_lr.predict_proba(X_meta)[:, 1]
        else:
            z = self.w_xgb * xgb_margin + self.w_dl * dl_logit + self.intercept
            return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(
        self,
        X_xgb: np.ndarray,
        cat_dl: torch.Tensor,
        num_dl: torch.Tensor,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        """
        End-to-end forward prediction given preprocessed inputs for XGBoost and Deep Learning.
        """
        if self.xgb_model is None or self.dl_model is None:
            raise ValueError("Both xgb_model and dl_model must be set.")

        xgb_margin = self.xgb_model.predict_margin(X_xgb)
        if hasattr(self.dl_model, "predict_logits"):
            dl_logit = self.dl_model.predict_logits(cat_dl, num_dl, device=device)
        else:
            dl_probs = self.dl_model.predict_proba(cat_dl, num_dl, device=device)
            dl_logit = np.log(np.clip(dl_probs, 1e-7, 1 - 1e-7) / (1 - np.clip(dl_probs, 1e-7, 1 - 1e-7)))

        return self.predict_fusion_proba(xgb_margin, dl_logit)

    def predict(
        self,
        X_xgb: np.ndarray,
        cat_dl: torch.Tensor,
        num_dl: torch.Tensor,
        threshold: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> np.ndarray:
        probs = self.predict_proba(X_xgb, cat_dl, num_dl, device=device)
        return (probs >= threshold).astype(int)

    def get_fusion_weights(self) -> Dict[str, float]:
        """Returns interpretable fusion weights for audit and reporting."""
        return {
            "w_xgb": self.w_xgb,
            "w_dl": self.w_dl,
            "intercept": self.intercept,
            "fusion_strategy": self.fusion_strategy,
        }

    def save(self, filepath: Path) -> None:
        """Saves hybrid model metadata and meta-weights."""
        import pickle

        filepath.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "fusion_strategy": self.fusion_strategy,
            "meta_l2": self.meta_l2,
            "random_seed": self.random_seed,
            "w_xgb": self.w_xgb,
            "w_dl": self.w_dl,
            "intercept": self.intercept,
            "is_fitted": self.is_fitted,
            "meta_lr": self.meta_lr,
        }
        with open(filepath, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(
        cls,
        filepath: Path,
        xgb_model: Optional[XGBoostModel] = None,
        dl_model: Optional[nn.Module] = None,
    ) -> "HybridFlightDelayModel":
        """Loads hybrid model meta-parameters from disk and links constituent models."""
        import pickle

        with open(filepath, "rb") as f:
            bundle = pickle.load(f)
        obj = cls(
            xgb_model=xgb_model,
            dl_model=dl_model,
            fusion_strategy=bundle.get("fusion_strategy", "gated_stacking"),
            meta_l2=bundle.get("meta_l2", 0.01),
            random_seed=bundle.get("random_seed", 42),
        )
        obj.w_xgb = bundle.get("w_xgb", 0.5)
        obj.w_dl = bundle.get("w_dl", 0.5)
        obj.intercept = bundle.get("intercept", 0.0)
        obj.is_fitted = bundle.get("is_fitted", True)
        obj.meta_lr = bundle.get("meta_lr", None)
        return obj
