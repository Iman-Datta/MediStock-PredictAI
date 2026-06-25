"""
feature_engineering.py
Shared feature engineering pipeline for training and inference.
"""

import pandas as pd
import numpy as np


CATEGORICALS = ["medicine_id", "medicine_name", "category", "season", "medicine_importance"]

FEATURES = [
    # raw
    "current_stock", "previous_week_sales", "sales_last_2_week",
    "sales_last_4_week_average", "flu_cases", "dengue_cases", "covid_cases",
    "supplier_delivery_days", "stock_received", "expiry_risk", "is_holiday_week",
    # engineered numeric
    "demand_lag1", "demand_lag2", "demand_lag4", "demand_lag8",
    "demand_roll4_mean", "demand_roll8_mean", "demand_roll4_std",
    "demand_trend", "stock_coverage", "disease_pressure",
    "reorder_urgency", "sales_momentum",
    "week_sin", "week_cos", "week_number", "year", "time_idx",
    # categoricals
    "medicine_id", "medicine_name", "category", "season", "medicine_importance",
]

IMPORTANCE_ORDER = ["Critical", "High", "Medium", "Low"]
IMPORTANCE_COLORS = {
    "Critical": "#F44336",
    "High":     "#FF9800",
    "Medium":   "#2196F3",
    "Low":      "#4CAF50",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input df must be sorted by medicine_id, year, week_number.
    Lag features are built per-medicine group (no leakage).
    """
    df = df.copy()
    df = df.sort_values(["medicine_id", "year", "week_number"]).reset_index(drop=True)

    # global time index
    df["time_idx"] = (df["year"] - 2023) * 52 + df["week_number"]

    # ── per-medicine lag features ─────────────────────────────────────────────
    grp = df.groupby("medicine_id")["next_week_demand"]
    df["demand_lag1"] = grp.shift(1)
    df["demand_lag2"] = grp.shift(2)
    df["demand_lag4"] = grp.shift(4)
    df["demand_lag8"] = grp.shift(8)

    # ── rolling stats (shift-1 to avoid leakage) ─────────────────────────────
    shifted = df.groupby("medicine_id")["next_week_demand"].shift(1)
    df["demand_roll4_mean"] = (
        shifted.groupby(df["medicine_id"])
               .transform(lambda x: x.rolling(4, min_periods=1).mean())
    )
    df["demand_roll8_mean"] = (
        shifted.groupby(df["medicine_id"])
               .transform(lambda x: x.rolling(8, min_periods=1).mean())
    )
    df["demand_roll4_std"] = (
        shifted.groupby(df["medicine_id"])
               .transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    )

    # ── derived features ──────────────────────────────────────────────────────
    df["demand_trend"]    = df["demand_roll4_mean"] - df["demand_roll8_mean"]
    df["stock_coverage"]  = df["current_stock"] / df["previous_week_sales"].clip(lower=1)
    df["disease_pressure"] = (
        df["flu_cases"] * 0.5 + df["dengue_cases"] * 0.3 + df["covid_cases"] * 0.2
    )
    df["reorder_urgency"] = df["supplier_delivery_days"] / df["stock_coverage"].clip(lower=0.1)
    df["sales_momentum"]  = df["previous_week_sales"] / df["sales_last_4_week_average"].clip(lower=1)

    # ── cyclical encoding ─────────────────────────────────────────────────────
    df["week_sin"] = np.sin(2 * np.pi * df["week_number"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_number"] / 52)

    # ── categorical dtypes (XGBoost native categorical) ───────────────────────
    for col in CATEGORICALS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def build_inference_row(inputs: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame ready for model inference.
    `inputs` must contain the fields listed in FEATURES (except derived ones).
    Missing lag/rolling fields are imputed from available sales signals.
    """
    row = inputs.copy()

    # impute lag fields from provided sales signals if absent
    sales_ref = row.get("previous_week_sales", row.get("sales_last_4_week_average", 50))

    for key, default in [
        ("demand_lag1",      row.get("previous_week_sales", sales_ref)),
        ("demand_lag2",      row.get("sales_last_2_week",   sales_ref)),
        ("demand_lag4",      row.get("sales_last_4_week_average", sales_ref)),
        ("demand_lag8",      sales_ref * 0.95),
        ("demand_roll4_mean", row.get("sales_last_4_week_average", sales_ref)),
        ("demand_roll8_mean", sales_ref * 0.97),
        ("demand_roll4_std",  sales_ref * 0.08),
        ("demand_trend",      0.0),
    ]:
        row.setdefault(key, default)

    # derived
    prev = max(row.get("previous_week_sales", 1), 1)
    row["stock_coverage"]  = row["current_stock"] / prev
    row["disease_pressure"] = (
        row.get("flu_cases", 0) * 0.5
        + row.get("dengue_cases", 0) * 0.3
        + row.get("covid_cases", 0) * 0.2
    )
    row["reorder_urgency"] = row["supplier_delivery_days"] / max(row["stock_coverage"], 0.1)
    avg = max(row.get("sales_last_4_week_average", 1), 1)
    row["sales_momentum"]  = row.get("previous_week_sales", avg) / avg

    wk = row.get("week_number", 1)
    row["week_sin"] = np.sin(2 * np.pi * wk / 52)
    row["week_cos"] = np.cos(2 * np.pi * wk / 52)
    row["time_idx"] = (row.get("year", 2025) - 2023) * 52 + wk

    df_row = pd.DataFrame([row])

    # categoricals
    for col in CATEGORICALS:
        if col in df_row.columns:
            df_row[col] = df_row[col].astype("category")

    return df_row[FEATURES]


def classify_stockout_risk(
    predicted_demand: float,
    current_stock: float,
    supplier_days: int,
    avg_daily_demand: float,
    low_thresh: float = 1.2,
    critical_thresh: float = 0.6,
) -> dict:
    """
    Classify stockout risk based on stock coverage relative to demand.
    Returns a dict with level, color, and description.
    """
    safety_stock = avg_daily_demand * supplier_days
    weeks_of_stock = current_stock / max(predicted_demand, 1)

    if weeks_of_stock <= critical_thresh or current_stock < safety_stock * 0.5:
        level, color, icon = "CRITICAL", "#F44336", "🔴"
    elif weeks_of_stock <= low_thresh or current_stock < safety_stock:
        level, color, icon = "LOW", "#FF9800", "🟠"
    elif weeks_of_stock <= 2.5:
        level, color, icon = "MODERATE", "#FDD835", "🟡"
    else:
        level, color, icon = "HEALTHY", "#4CAF50", "🟢"

    recommended_order = max(0, round(
        predicted_demand * (1 + supplier_days / 7) - current_stock
    ))

    return {
        "level": level,
        "color": color,
        "icon": icon,
        "weeks_of_stock": round(weeks_of_stock, 1),
        "safety_stock": round(safety_stock),
        "recommended_order": recommended_order,
    }
