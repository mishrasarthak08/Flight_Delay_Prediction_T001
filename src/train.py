"""
Training routines for Logistic Regression, XGBoost, and PyTorch Embedding MLP.
Enforces validation monitoring, early stopping, and checkpointing.
"""

from typing import Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.config import (
    ModelHyperparams,
    MODELS_DIR,
    TARGET_COL,
    CATEGORICAL_FEATURES,
    NUMERIC_SCHEDULE_FEATURES,
    HISTORICAL_FEATURES,
)
from src.preprocessing import (
    FlightTabularPreprocessor,
    PyTorchPreprocessor,
    FlightDataset,
)
from src.models import (
    LogisticRegressionBaseline,
    XGBoostModel,
    FlightEmbeddingMLP,
    FlightDeepResNet,
    HybridFlightDelayModel,
)


def train_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    output_dir: Path = MODELS_DIR,
    numeric_features=None,
) -> Tuple[
    LogisticRegressionBaseline, FlightTabularPreprocessor, np.ndarray, np.ndarray
]:
    """
    Trains Logistic Regression baseline using scikit-learn preprocessing pipeline.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = FlightTabularPreprocessor(
        numeric_features=numeric_features, sparse_output=False
    )

    feature_cols = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )
    X_train = np.ascontiguousarray(
        preprocessor.fit_transform(train_df[feature_cols]), dtype=np.float32
    )
    y_train = np.ascontiguousarray(train_df[TARGET_COL].values, dtype=np.int32)

    X_val = np.ascontiguousarray(
        preprocessor.transform(val_df[feature_cols]), dtype=np.float32
    )

    model = LogisticRegressionBaseline(hyperparams=hyperparams, random_seed=random_seed)
    model.fit(X_train, y_train)

    # Save artifacts
    preprocessor.save(output_dir / "logistic_regression_preprocessor.pkl")
    import pickle

    with open(output_dir / "logistic_regression.pkl", "wb") as f:
        pickle.dump(model, f)

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    return model, preprocessor, train_probs, val_probs


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: FlightTabularPreprocessor,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    output_dir: Path = MODELS_DIR,
) -> Tuple[XGBoostModel, np.ndarray, np.ndarray]:
    """
    Trains XGBoost model with early stopping on validation split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )

    X_train = np.ascontiguousarray(
        preprocessor.transform(train_df[feature_cols]), dtype=np.float32
    )
    y_train = np.ascontiguousarray(train_df[TARGET_COL].values, dtype=np.float32)

    X_val = np.ascontiguousarray(
        preprocessor.transform(val_df[feature_cols]), dtype=np.float32
    )
    y_val = np.ascontiguousarray(val_df[TARGET_COL].values, dtype=np.float32)

    print(f"     Fitting XGBoost hist on {X_train.shape}...")
    model = XGBoostModel(hyperparams=hyperparams, random_seed=random_seed)
    model.fit(X_train, y_train, X_val, y_val)
    print(f"     XGBoost fitted successfully! Rounds: {model.model.n_estimators}")

    # Save artifact
    import pickle

    with open(output_dir / "xgboost.pkl", "wb") as f:
        pickle.dump(model, f)

    train_probs = model.predict_proba(X_train)
    val_probs = model.predict_proba(X_val)
    return model, train_probs, val_probs


def train_pytorch_mlp(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    output_dir: Path = MODELS_DIR,
    select_checkpoint: bool = True,
    numeric_features=None,
) -> Tuple[FlightEmbeddingMLP, PyTorchPreprocessor, np.ndarray, np.ndarray]:
    """
    Trains PyTorch Embedding MLP with early stopping on validation loss.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if device is None:
        device = torch.device("cpu")

    # Fit PyTorch Preprocessor strictly on training data
    preprocessor = PyTorchPreprocessor(numeric_features=numeric_features)
    feature_cols = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )
    preprocessor.fit(train_df[feature_cols])

    # Convert to Tensors
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    c_train, n_train, yt_train = preprocessor.transform(train_df[feature_cols], y_train)
    c_val, n_val, yt_val = preprocessor.transform(val_df[feature_cols], y_val)

    train_dataset = FlightDataset(c_train, n_train, yt_train)
    val_dataset = FlightDataset(c_val, n_val, yt_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=False,
    )

    # Model definition
    num_carriers = preprocessor.vocab_sizes["carrier"]
    num_origins = preprocessor.vocab_sizes["origin"]
    num_dests = preprocessor.vocab_sizes["dest"]
    num_numeric = len(preprocessor.numeric_features)

    model = FlightEmbeddingMLP(
        num_carriers=num_carriers,
        num_origins=num_origins,
        num_dests=num_dests,
        num_numeric_features=num_numeric,
        carrier_emb_dim=hyperparams.mlp_carrier_emb_dim,
        origin_emb_dim=hyperparams.mlp_origin_emb_dim,
        dest_emb_dim=hyperparams.mlp_dest_emb_dim,
        hidden_dims=hyperparams.mlp_hidden_dims,
        dropout=hyperparams.mlp_dropout,
    )
    model.to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyperparams.mlp_learning_rate,
        weight_decay=hyperparams.mlp_weight_decay,
    )

    best_val_loss = float("inf")
    best_state_dict = None
    patience = hyperparams.mlp_early_stopping_patience
    patience_counter = 0

    history = []
    for epoch in range(1, hyperparams.mlp_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        train_loss /= len(train_dataset)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_cat, batch_num)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_list.append(probs)

        val_loss /= len(val_dataset)
        val_probs_arr = np.concatenate(val_preds_list)
        val_prauc = average_precision_score(y_val, val_probs_arr)
        history.append(
            {
                "epoch": epoch,
                "train_log_loss": float(train_loss),
                "monitor_log_loss": float(val_loss),
                "monitor_average_precision": float(val_prauc),
            }
        )
        print(
            f"     [Epoch {epoch}/{hyperparams.mlp_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_prauc:.4f}"
        )

        if not select_checkpoint:
            continue
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save artifacts
    with open(output_dir / "pytorch_training_history.json", "w") as f:
        json.dump(
            {
                "checkpoint_selection": (
                    "training_internal_monitor"
                    if select_checkpoint
                    else "fixed_epochs_no_monitor_selection"
                ),
                "history": history,
            },
            f,
            indent=2,
        )
    preprocessor.save(output_dir / "pytorch_preprocessor.pkl")
    model.save_checkpoint(
        output_dir / "pytorch_mlp.pt",
        config_dict={
            "best_val_loss": float(best_val_loss) if select_checkpoint else None,
            "epochs_run": epoch,
            "random_seed": random_seed,
        },
    )

    # Compute final predictions on CPU
    model.to(torch.device("cpu"))
    train_probs = model.predict_proba(c_train, n_train)
    val_probs = model.predict_proba(c_val, n_val)
    return model, preprocessor, train_probs, val_probs


def train_pytorch_deep_resnet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    device: Optional[torch.device] = None,
    output_dir: Path = MODELS_DIR,
    select_checkpoint: bool = True,
    numeric_features=None,
) -> Tuple[FlightDeepResNet, PyTorchPreprocessor, np.ndarray, np.ndarray]:
    """
    Trains Phase 2 Deep Tabular ResNet with residual skip connections,
    cosine learning rate annealing, weight decay, dropout, and gradient clipping.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if device is None:
        device = torch.device("cpu")

    preprocessor = PyTorchPreprocessor(numeric_features=numeric_features)
    feature_cols = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )
    preprocessor.fit(train_df[feature_cols])

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values

    c_train, n_train, yt_train = preprocessor.transform(train_df[feature_cols], y_train)
    c_val, n_val, yt_val = preprocessor.transform(val_df[feature_cols], y_val)

    train_dataset = FlightDataset(c_train, n_train, yt_train)
    val_dataset = FlightDataset(c_val, n_val, yt_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams.mlp_batch_size,
        shuffle=False,
    )

    num_carriers = preprocessor.vocab_sizes["carrier"]
    num_origins = preprocessor.vocab_sizes["origin"]
    num_dests = preprocessor.vocab_sizes["dest"]
    num_numeric = len(preprocessor.numeric_features)

    model = FlightDeepResNet(
        num_carriers=num_carriers,
        num_origins=num_origins,
        num_dests=num_dests,
        num_numeric_features=num_numeric,
        carrier_emb_dim=hyperparams.mlp_carrier_emb_dim,
        origin_emb_dim=hyperparams.mlp_origin_emb_dim,
        dest_emb_dim=hyperparams.mlp_dest_emb_dim,
        hidden_dim=hyperparams.resnet_hidden_dim,
        num_blocks=hyperparams.resnet_num_blocks,
        dropout=hyperparams.resnet_dropout,
        use_layernorm=hyperparams.resnet_use_layernorm,
    )
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams.mlp_learning_rate,
        weight_decay=hyperparams.mlp_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hyperparams.mlp_epochs, eta_min=1e-5
    )

    best_val_loss = float("inf")
    best_state_dict = None
    patience = hyperparams.mlp_early_stopping_patience
    patience_counter = 0

    history = []
    print(f"     Fitting Deep Tabular ResNet ({hyperparams.resnet_num_blocks} blocks, {hyperparams.resnet_hidden_dim}d) on {device}...")
    for epoch in range(1, hyperparams.mlp_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_cat, batch_num, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_num = batch_num.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_cat, batch_num)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)

        scheduler.step()
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_preds_list = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_num = batch_num.to(device)
                batch_y = batch_y.to(device)

                logits = model(batch_cat, batch_num)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds_list.append(probs)

        val_loss /= len(val_dataset)
        val_probs_arr = np.concatenate(val_preds_list)
        val_prauc = average_precision_score(y_val, val_probs_arr)
        history.append(
            {
                "epoch": epoch,
                "train_log_loss": float(train_loss),
                "monitor_log_loss": float(val_loss),
                "monitor_average_precision": float(val_prauc),
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        print(
            f"     [Epoch {epoch}/{hyperparams.mlp_epochs}] ResNet Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_prauc:.4f}"
        )

        if not select_checkpoint:
            continue
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"     Early stopping triggered at epoch {epoch}.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    with open(output_dir / "pytorch_resnet_training_history.json", "w") as f:
        json.dump(
            {
                "checkpoint_selection": (
                    "training_internal_monitor"
                    if select_checkpoint
                    else "fixed_epochs_no_monitor_selection"
                ),
                "history": history,
            },
            f,
            indent=2,
        )
    preprocessor.save(output_dir / "pytorch_preprocessor.pkl")
    model.save_checkpoint(
        output_dir / "pytorch_deep_resnet.pt",
        config_dict={
            "best_val_loss": float(best_val_loss) if select_checkpoint else None,
            "epochs_run": epoch,
            "random_seed": random_seed,
        },
    )

    model.to(torch.device("cpu"))
    train_probs = model.predict_proba(c_train, n_train)
    val_probs = model.predict_proba(c_val, n_val)
    return model, preprocessor, train_probs, val_probs


def train_hybrid_model(
    xgb_model: XGBoostModel,
    xgb_prep: FlightTabularPreprocessor,
    dl_model: nn.Module,
    dl_prep: PyTorchPreprocessor,
    fitted_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hyperparams: ModelHyperparams = ModelHyperparams(),
    random_seed: int = 42,
    output_dir: Path = MODELS_DIR,
    device: Optional[torch.device] = None,
) -> Tuple[HybridFlightDelayModel, np.ndarray, np.ndarray]:
    """
    Trains the Phase 2 Hybrid ML-DL Model:
    1. Extracts tree margins from XGBoost on fitted and validation data.
    2. Extracts deep network logits from DL model on fitted and validation data.
    3. Fits the regularized logit-space stacking meta-learner on the validation block.
    4. Serializes the fitted hybrid model and its interpretability weights.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cpu")

    feature_cols = (
        CATEGORICAL_FEATURES + NUMERIC_SCHEDULE_FEATURES + HISTORICAL_FEATURES
    )

    # 1. Transform inputs for XGBoost
    X_xgb_train = xgb_prep.transform(fitted_df[feature_cols])
    X_xgb_val = xgb_prep.transform(val_df[feature_cols])

    # 2. Transform inputs for Deep Learning
    c_train, n_train, _ = dl_prep.transform(fitted_df[feature_cols])
    c_val, n_val, _ = dl_prep.transform(val_df[feature_cols])

    # 3. Compute raw margins / logits
    xgb_margin_train = xgb_model.predict_margin(X_xgb_train)
    xgb_margin_val = xgb_model.predict_margin(X_xgb_val)

    if hasattr(dl_model, "predict_logits"):
        dl_logit_train = dl_model.predict_logits(c_train, n_train, device=device)
        dl_logit_val = dl_model.predict_logits(c_val, n_val, device=device)
    else:
        dl_probs_train = dl_model.predict_proba(c_train, n_train, device=device)
        dl_probs_val = dl_model.predict_proba(c_val, n_val, device=device)
        dl_logit_train = np.log(np.clip(dl_probs_train, 1e-7, 1 - 1e-7) / (1 - np.clip(dl_probs_train, 1e-7, 1 - 1e-7)))
        dl_logit_val = np.log(np.clip(dl_probs_val, 1e-7, 1 - 1e-7) / (1 - np.clip(dl_probs_val, 1e-7, 1 - 1e-7)))

    # 4. Instantiate and fit Hybrid model
    hybrid = HybridFlightDelayModel(
        xgb_model=xgb_model,
        dl_model=dl_model,
        fusion_strategy=hyperparams.hybrid_fusion_strategy,
        meta_l2=hyperparams.hybrid_meta_l2,
        random_seed=random_seed,
    )
    print(f"     Fitting Hybrid fusion ({hyperparams.hybrid_fusion_strategy}, L2={hyperparams.hybrid_meta_l2})...")
    hybrid.fit_fusion(xgb_margin_val, dl_logit_val, val_df[TARGET_COL].to_numpy())
    weights = hybrid.get_fusion_weights()
    print(f"     Hybrid fusion fitted! Weights: w_xgb={weights['w_xgb']:.4f}, w_dl={weights['w_dl']:.4f}, b={weights['intercept']:.4f}")

    # 5. Save hybrid model
    hybrid.save(output_dir / "hybrid_model.pkl")

    train_probs = hybrid.predict_fusion_proba(xgb_margin_train, dl_logit_train)
    val_probs = hybrid.predict_fusion_proba(xgb_margin_val, dl_logit_val)

    return hybrid, train_probs, val_probs
