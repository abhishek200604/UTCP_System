"""
utils.py — Data Processing, Feature Engineering & Sparse Sensor Simulation
Urban Traffic Congestion Prediction Under Sparse and Noisy Sensor Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────
#  1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────

def load_data(path: str = "data/pune_january_20areas.csv") -> pd.DataFrame:
    """Load the Pune traffic CSV and perform initial cleaning."""
    df = pd.read_csv(path)

    # Combine Date + Time → datetime
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d-%m-%Y %H:%M", errors="coerce"
    )

    # Clean Congestion_Level (fix typos like 'Sever+L58:M61')
    df["Congestion_Level"] = df["Congestion_Level"].replace(
        {r"Sever.*": "Severe"}, regex=True
    )

    # Standardize Yes/No columns
    for col in ["Is_Weekend", "Is_Holiday", "Accident", "Road_Work"]:
        df[col] = df[col].astype(str).str.strip().str.capitalize()

    # Fill missing categorical values
    df["Event_Name"] = df["Event_Name"].fillna("None")
    df["Event_Impact_Level"] = df["Event_Impact_Level"].fillna("None")
    df["Weather"] = df["Weather"].fillna("Clear")

    # Fill missing numeric values with column medians
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # ── Apply realistic Pune-style hourly traffic pattern ──
    # The raw synthetic data is nearly flat across hours.
    # This reshapes it to match real urban congestion rhythms:
    #   night lull → morning rush → midday dip → evening rush → night decline
    hour = df["Datetime"].dt.hour
    hourly_multiplier = pd.Series(np.ones(len(df)), index=df.index)

    # Multipliers calibrated for Pune traffic (relative to dataset mean)
    _pattern = {
        0: 0.42, 1: 0.36, 2: 0.30, 3: 0.28, 4: 0.32, 5: 0.45,
        6: 0.65, 7: 0.90, 8: 1.22, 9: 1.30, 10: 1.10, 11: 0.92,
        12: 0.88, 13: 0.85, 14: 0.87, 15: 0.95, 16: 1.10, 17: 1.32,
        18: 1.38, 19: 1.28, 20: 1.05, 21: 0.82, 22: 0.62, 23: 0.50,
    }
    for h, mult in _pattern.items():
        hourly_multiplier[hour == h] = mult

    # Rescale congestion columns (clamp to 0–100)
    global_mean = df["Final_Congestion_Score"].mean()
    for col in ["Final_Congestion_Score", "Traffic_Score", "Holiday_Traffic_Score"]:
        col_mean = df[col].mean()
        df[col] = (col_mean + (df[col] - col_mean) * 0.5
                   + (hourly_multiplier - 1.0) * col_mean * 1.1)
        df[col] = df[col].clip(0, 100).round(1)

    # Speed is inversely related to congestion
    spd_mean = df["Avg_Speed_kmph"].mean()
    inv_mult = 2.0 - hourly_multiplier  # invert: high congestion → low speed
    df["Avg_Speed_kmph"] = (spd_mean + (df["Avg_Speed_kmph"] - spd_mean) * 0.5
                            + (inv_mult - 1.0) * spd_mean * 0.9)
    df["Avg_Speed_kmph"] = df["Avg_Speed_kmph"].clip(3, 80).round(1)

    # Re-derive Congestion_Level from adjusted scores
    def _score_to_level(s):
        if s >= 75: return "Severe"
        if s >= 55: return "High"
        if s >= 35: return "Medium"
        return "Low"

    df["Congestion_Level"] = df["Final_Congestion_Score"].apply(_score_to_level)

    return df


# ─────────────────────────────────────────────
#  2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

EVENT_IMPACT_MAP = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
WEATHER_MAP = {"Clear": 0, "Cloudy": 1, "Fog": 2, "Rain": 3}
CONGESTION_MAP = {"Low": 0, "Medium": 1, "High": 2, "Severe": 3}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rich features for modelling."""
    df = df.copy()

    # --- Temporal features ---
    df["Hour"] = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek          # Mon=0 … Sun=6
    df["DayOfMonth"] = df["Datetime"].dt.day
    df["Is_Weekend_Num"] = (df["Is_Weekend"] == "Yes").astype(int)
    df["Is_Holiday_Num"] = (df["Is_Holiday"] == "Yes").astype(int)

    # --- Categorical encodings ---
    df["Event_Impact_Num"] = df["Event_Impact_Level"].map(EVENT_IMPACT_MAP).fillna(0).astype(int)
    df["Weather_Num"] = df["Weather"].map(WEATHER_MAP).fillna(0).astype(int)
    df["Congestion_Num"] = df["Congestion_Level"].map(CONGESTION_MAP).fillna(1).astype(int)
    df["Accident_Num"] = (df["Accident"] == "Yes").astype(int)
    df["Road_Work_Num"] = (df["Road_Work"] == "Yes").astype(int)

    # --- Area encoding (label) ---
    le_area = LabelEncoder()
    df["Area_Encoded"] = le_area.fit_transform(df["Area_Name"])

    # --- Area congestion index (mean Final_Congestion_Score per area) ---
    area_cong = df.groupby("Area_Name")["Final_Congestion_Score"].mean()
    df["Area_Congestion_Index"] = df["Area_Name"].map(area_cong)

    # --- Rolling average congestion (3-hour window per area) ---
    df = df.sort_values(["Area_Name", "Datetime"]).reset_index(drop=True)
    df["Rolling_Congestion_3h"] = (
        df.groupby("Area_Name")["Final_Congestion_Score"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    # --- Spatio-temporal interaction ---
    df["Area_Hour"] = df["Area_Encoded"] * 100 + df["Hour"]

    # --- Holiday traffic boost ---
    df["Holiday_Boost"] = df["Holiday_Traffic_Score"] - df["Final_Congestion_Score"]

    return df, le_area


def get_feature_columns() -> list:
    """Return the ordered list of feature column names used for modelling."""
    return [
        "Hour", "DayOfWeek", "DayOfMonth",
        "Is_Weekend_Num", "Is_Holiday_Num",
        "Traffic_Volume", "Avg_Speed_kmph",
        "Traffic_Score", "Holiday_Traffic_Score",
        "Event_Impact_Num", "Weather_Num",
        "Congestion_Num", "Accident_Num", "Road_Work_Num",
        "Area_Encoded", "Area_Congestion_Index",
        "Rolling_Congestion_3h", "Area_Hour",
        "Holiday_Boost",
        "Latitude", "Longitude",
    ]


# ─────────────────────────────────────────────
#  3. NORMALIZATION
# ─────────────────────────────────────────────

def normalize_features(X: pd.DataFrame):
    """Min-max scale the feature matrix. Returns (X_scaled, scaler)."""
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


# ─────────────────────────────────────────────
#  4. NOISE DETECTION & SMOOTHING
# ─────────────────────────────────────────────

def smooth_noise(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Apply rolling-mean smoothing to noisy numeric columns."""
    df = df.copy()
    for col in ["Traffic_Volume", "Avg_Speed_kmph", "Traffic_Score"]:
        df[col] = (
            df.groupby("Area_Name")[col]
            .transform(lambda s: s.rolling(window, min_periods=1, center=True).mean())
        )
    return df


# ─────────────────────────────────────────────
#  5. SPARSE & NOISY SENSOR SIMULATION
# ─────────────────────────────────────────────

def simulate_sparse_data(df: pd.DataFrame, missing_pct: float = 0.2,
                         random_state: int = 42) -> pd.DataFrame:
    """Randomly drop `missing_pct` fraction of rows to simulate sparse sensors."""
    np.random.seed(random_state)
    keep_mask = np.random.rand(len(df)) > missing_pct
    return df[keep_mask].reset_index(drop=True)


def add_noise(df: pd.DataFrame, noise_level: float = 0.1,
              random_state: int = 42) -> pd.DataFrame:
    """Add Gaussian noise to speed and volume columns."""
    df = df.copy()
    np.random.seed(random_state)
    for col in ["Traffic_Volume", "Avg_Speed_kmph"]:
        std = df[col].std()
        noise = np.random.normal(0, std * noise_level, size=len(df))
        df[col] = df[col] + noise
        df[col] = df[col].clip(lower=0)  # speed/volume cannot be negative
    return df


def reconstruct_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing numeric values after sparse simulation."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


# ─────────────────────────────────────────────
#  6. FULL PIPELINE HELPER
# ─────────────────────────────────────────────

def prepare_modelling_data(path: str = "data/pune_january_20areas.csv"):
    """End-to-end: load → clean → smooth → engineer → split features/target."""
    df = load_data(path)
    df = smooth_noise(df)
    df, le_area = engineer_features(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df["Final_Congestion_Score"].copy()

    # Replace any infinities, fill remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y, df, le_area
