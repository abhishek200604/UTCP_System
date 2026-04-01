"""
train_model.py — ML Training Pipeline
Urban Traffic Congestion Prediction Under Sparse and Noisy Sensor Data

Trains Linear Regression, Random Forest, and Gradient Boosting regressors
on the Pune traffic dataset. Evaluates with MAE / RMSE / R², saves the
best model and a JSON metrics report.
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from utils import prepare_modelling_data, normalize_features

# ───────── configuration ─────────
DATA_PATH = "data/pune_january_20areas.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "congestion_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ───────── ANSI colour helpers ─────────
class C:
    """ANSI escape-code shortcuts for coloured terminal output."""
    RST   = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAG   = "\033[95m"
    RED   = "\033[91m"
    BLUE  = "\033[94m"
    WHITE = "\033[97m"


def _emoji_extra(text):
    """Count extra columns emojis consume (each emoji = 2 cols, Python counts 1)."""
    import unicodedata
    extra = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ('W', 'F'):
            extra += 1
        elif ord(ch) > 0xFFFF:  # supplementary plane (most emojis)
            extra += 1
    return extra


def _cbox(text, width):
    """Center text in a box cell, accounting for emoji double-width."""
    pad = width - len(text) - _emoji_extra(text)
    left = pad // 2
    right = pad - left
    return ' ' * left + text + ' ' * right


def _header():
    W = 56
    print()
    print(f"  {C.CYAN}{C.BOLD}╔{'═' * W}╗{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{_cbox('UTCP  —  Model Training Pipeline', W)}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{_cbox('Urban Traffic Congestion Prediction', W)}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.CYAN}{C.BOLD}╚{'═' * W}╝{C.RST}")
    print()


def _step(num, total, emoji, msg):
    print(f"  {C.BLUE}{C.BOLD}[{num}/{total}]{C.RST}  {emoji}  {C.WHITE}{C.BOLD}{msg}{C.RST}")


def _info(msg):
    print(f"        {C.DIM}▸ {msg}{C.RST}")


def _r2_color(val):
    if val >= 0.99:
        return C.GREEN + C.BOLD
    elif val >= 0.9:
        return C.GREEN
    elif val >= 0.7:
        return C.YELLOW
    return C.RED


def evaluate(model, X_test, y_test) -> dict:
    """Return MAE, RMSE, R² for a fitted model."""
    preds = model.predict(X_test)
    return {
        "MAE": round(mean_absolute_error(y_test, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "R2": round(r2_score(y_test, preds), 4),
    }


def train_all_models():
    """Full training pipeline: process data, train 3 models, compare, save best."""
    t0 = time.time()
    _header()

    # ── Step 1: Load & prepare ──
    _step(1, 5, "📂", "Loading & processing dataset …")
    X, y, df, le_area = prepare_modelling_data(DATA_PATH)
    _info(f"Rows: {C.CYAN}{X.shape[0]:,}{C.RST}{C.DIM}  ·  Features: {C.CYAN}{X.shape[1]}{C.RST}")
    _info(f"Target range: {C.YELLOW}{y.min():.1f}{C.RST}{C.DIM} – {C.YELLOW}{y.max():.1f}{C.RST}")
    print()

    # ── Step 2: Normalize ──
    _step(2, 5, "⚙️ ", "Normalizing features (MinMaxScaler) …")
    X_scaled, scaler = normalize_features(X)
    _info("All features scaled to [0, 1]")
    print()

    # ── Step 3: Split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _step(3, 5, "✂️ ", "Train / Test split")
    _info(f"Train: {C.GREEN}{len(X_train):,}{C.RST}{C.DIM}  ·  Test: {C.YELLOW}{len(X_test):,}{C.RST}{C.DIM}  ({TEST_SIZE:.0%} held out)")
    print()

    # ── Step 4: Train ──
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, max_depth=25, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.9, random_state=RANDOM_STATE
        ),
    }

    _step(4, 5, "🧠", "Training models …")
    print()

    results = {}
    model_emojis = {
        "Linear Regression": "📏",
        "Random Forest": "🌲",
        "Gradient Boosting": "🚀",
    }

    # Table header
    print(f"        {C.DIM}{'─' * 70}{C.RST}")
    print(f"        {C.BOLD}{'  Model':<24}{'R²':>10}{'MAE':>12}{'RMSE':>12}   Status{C.RST}")
    print(f"        {C.DIM}{'─' * 70}{C.RST}")

    for name, model in models.items():
        emoji = model_emojis.get(name, "🔧")
        t1 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t1
        metrics = evaluate(model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics}

        r2c = _r2_color(metrics["R2"])
        status = (
            f"{C.GREEN}✅ {elapsed:.1f}s{C.RST}"
            if metrics["R2"] >= 0.9
            else f"{C.YELLOW}⚠️  {elapsed:.1f}s{C.RST}"
        )
        print(
            f"        {emoji} {C.WHITE}{name:<22}{C.RST}"
            f"{r2c}{metrics['R2']:>9.4f}{C.RST}"
            f"{C.CYAN}{metrics['MAE']:>12.4f}{C.RST}"
            f"{C.CYAN}{metrics['RMSE']:>12.4f}{C.RST}"
            f"   {status}"
        )

    print(f"        {C.DIM}{'─' * 70}{C.RST}")
    print()

    # ── Step 5: Save ──
    best_name = max(results, key=lambda k: results[k]["metrics"]["R2"])
    best_model = results[best_name]["model"]
    best_r2 = results[best_name]["metrics"]["R2"]

    _step(5, 5, "💾", "Saving artifacts …")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    metrics_out = {name: results[name]["metrics"] for name in results}
    metrics_out["best_model"] = best_name
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics_out, f, indent=2)

    _info(f"Model  → {C.CYAN}{MODEL_FILE}{C.RST}")
    _info(f"Scaler → {C.CYAN}{SCALER_FILE}{C.RST}")
    _info(f"Report → {C.CYAN}{METRICS_FILE}{C.RST}")
    print()

    # ── Summary banner ──
    W = 56
    elapsed_total = time.time() - t0
    print(f"  {C.GREEN}{C.BOLD}╔{'═' * W}╗{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}║{_cbox('TRAINING COMPLETE', W)}║{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}║{_cbox(f'Best: {best_name}  (R² = {best_r2:.4f})', W)}║{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}║{_cbox(f'{elapsed_total:.1f}s total', W)}║{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}║{' ' * W}║{C.RST}")
    print(f"  {C.GREEN}{C.BOLD}╚{'═' * W}╝{C.RST}")
    print()
    print(f"  {C.MAG}💡 Run the dashboard:{C.RST}  {C.CYAN}streamlit run app.py{C.RST}")
    print()

    return results


if __name__ == "__main__":
    train_all_models()
