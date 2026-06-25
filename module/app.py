"""
app.py — Smart Pharmacy AI Inventory Management
Run: streamlit run app.py
"""

import json
import io
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from feature_engineering import (
    FEATURES, CATEGORICALS, IMPORTANCE_COLORS, IMPORTANCE_ORDER,
    engineer_features, build_inference_row, classify_stockout_risk,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Pharmacy AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  /* ── base ── */
  [data-testid="stAppViewContainer"] { background: #0D1117; }
  [data-testid="stSidebar"]          { background: #161B22; border-right: 1px solid #21262D; }
  .block-container { padding: 1.5rem 2rem 3rem; }

  /* ── typography ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #E6EDF3; }
  h1 { font-size: 1.6rem !important; font-weight: 700 !important; letter-spacing: -.3px; }
  h2 { font-size: 1.15rem !important; font-weight: 600 !important; color: #8B949E !important; }
  h3 { font-size: 1rem !important; font-weight: 600 !important; }
  label { color: #8B949E !important; font-size: .82rem !important; }

  /* ── cards ── */
  .card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: .75rem;
  }
  .card-tight { padding: .7rem 1rem; }

  /* ── metric pills ── */
  .metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: .8rem; }
  .metric-pill {
    background: #21262D;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: .55rem 1rem;
    min-width: 110px;
    flex: 1;
  }
  .metric-pill .label { font-size: .72rem; color: #8B949E; text-transform: uppercase;
                         letter-spacing: .05em; margin-bottom: 2px; }
  .metric-pill .value { font-size: 1.4rem; font-weight: 700; color: #E6EDF3; }
  .metric-pill .sub   { font-size: .72rem; color: #8B949E; margin-top: 1px; }

  /* ── risk banners ── */
  .risk-banner {
    border-radius: 8px;
    padding: .65rem 1rem;
    font-weight: 700;
    font-size: .95rem;
    display: flex;
    align-items: center;
    gap: .5rem;
    margin: .5rem 0;
  }

  /* ── tables ── */
  .stDataFrame { background: #161B22 !important; }
  thead th { background: #21262D !important; color: #8B949E !important; font-size: .78rem !important; }
  tbody td { font-size: .82rem !important; }

  /* ── sidebar ── */
  .sidebar-section {
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #00BFA5;
    margin: 1rem 0 .4rem;
    font-weight: 700;
  }

  /* ── tabs ── */
  [data-testid="stTabs"] button {
    font-size: .82rem !important;
    font-weight: 600 !important;
    color: #8B949E !important;
  }
  [data-testid="stTabs"] button[aria-selected="true"] {
    color: #00BFA5 !important;
    border-bottom-color: #00BFA5 !important;
  }

  /* ── buttons ── */
  .stButton > button {
    background: #00BFA5 !important;
    color: #0D1117 !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
    padding: .45rem 1.2rem !important;
  }
  .stButton > button:hover { background: #00D4B8 !important; }

  /* ── misc ── */
  .teal { color: #00BFA5; }
  .muted { color: #8B949E; font-size: .82rem; }
  hr.divider { border: none; border-top: 1px solid #21262D; margin: 1rem 0; }
  .align-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: .75rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model("xgb_demand_model.json")
    return m


@st.cache_data
def load_dataset():
    df = pd.read_csv("smart_pharmacy_dataset.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#161B22",
    font=dict(family="Inter", color="#E6EDF3", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    xaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#21262D", linecolor="#30363D", tickfont=dict(size=11)),
)


def apply_theme(fig, **extra):
    fig.update_layout(**PLOTLY_LAYOUT, **extra)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "## 💊 Smart Pharmacy AI\n"
        "<span class='muted'>Demand Forecasting & Stockout Prevention</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Risk Thresholds</div>", unsafe_allow_html=True)
    critical_thresh = st.slider("Critical stockout (weeks)", 0.2, 1.5, 0.6, 0.1,
                                help="Weeks of stock below which risk is CRITICAL")
    low_thresh      = st.slider("Low stockout (weeks)",      0.5, 3.0, 1.2, 0.1,
                                help="Weeks of stock below which risk is LOW")

    st.markdown("<div class='sidebar-section'>Reorder Buffer</div>", unsafe_allow_html=True)
    reorder_buffer  = st.slider("Extra buffer (%)", 0, 50, 10, 5,
                                help="Safety margin added on top of forecast for reorder calculation")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Model</div>", unsafe_allow_html=True)
    try:
        mdl = load_model()
        n_features = len(mdl.get_booster().feature_names)
        n_trees    = mdl.n_estimators
        st.markdown(f"""
        <div class='card card-tight'>
          <div class='muted'>XGBoost Regressor</div>
          <div style='margin-top:.4rem;font-size:.8rem;color:#E6EDF3'>
            <b>{n_features}</b> features &nbsp;·&nbsp; <b>{n_trees}</b> estimators
          </div>
          <div style='margin-top:.3rem;font-size:.75rem;color:#00BFA5'>R² 0.923 on 2025 holdout</div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model load error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:.75rem;margin-bottom:1rem'>
  <span style='font-size:2rem'>💊</span>
  <div>
    <h1 style='margin:0'>Smart Pharmacy AI Inventory</h1>
    <span class='muted'>XGBoost demand forecasting · Stockout risk classification · Priority reorder engine</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_dashboard, tab_inspect = st.tabs([
    "🔮 Single Prediction",
    "📋 Batch Forecast",
    "📊 Dashboard",
    "🔬 Model Inspector",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Enter Medicine Details")

    try:
        model = load_model()
        df_raw = load_dataset()
        medicine_names = sorted(df_raw["medicine_name"].unique().tolist())
        categories     = sorted(df_raw["category"].unique().tolist())
        med_ids        = sorted(df_raw["medicine_id"].unique().tolist())
    except Exception as e:
        st.error(f"Could not load model or data: {e}")
        st.stop()

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Medicine Identity**")
        med_name   = st.selectbox("Medicine Name", medicine_names, key="sp_name")
        # auto-fill medicine_id and category from dataset
        med_row    = df_raw[df_raw["medicine_name"] == med_name].iloc[0]
        med_id     = med_row["medicine_id"]
        category   = med_row["category"]
        med_imp    = med_row["medicine_importance"]
        st.markdown(
            f"<span class='muted'>ID: <b>{med_id}</b> &nbsp;·&nbsp; "
            f"Category: <b>{category}</b> &nbsp;·&nbsp; "
            f"Importance: <b style='color:{IMPORTANCE_COLORS.get(med_imp, '#E6EDF3')}'>{med_imp}</b></span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Time & Context**")
        c1, c2 = st.columns(2)
        with c1:
            year     = st.selectbox("Year", [2023, 2024, 2025, 2026], index=2)
            season   = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Normal"])
        with c2:
            week_num = st.slider("Week Number", 1, 52, 26)
            is_hol   = st.checkbox("Holiday Week")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Disease Pressure**")
        c1, c2, c3 = st.columns(3)
        flu    = c1.number_input("Flu Cases",    0, 600, 120, step=10)
        dengue = c2.number_input("Dengue Cases", 0, 500, 45,  step=5)
        covid  = c3.number_input("Covid Cases",  0, 200, 30,  step=5)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Inventory Status**")
        curr_stock = st.number_input("Current Stock (units)", 0, 5000, 150, step=10)
        c1, c2 = st.columns(2)
        prev_sales = c1.number_input("Last Week Sales",   1, 500, 55, step=5)
        sales_2w   = c2.number_input("Sales 2 Weeks Ago", 1, 500, 52, step=5)
        sales_4avg = st.number_input("4-Week Avg Sales", 1.0, 500.0, 53.0, step=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Supply Chain**")
        c1, c2 = st.columns(2)
        sup_days    = c1.number_input("Supplier Lead (days)", 1, 20, 5)
        stock_recv  = c2.number_input("Stock Received",       0, 700, 60, step=10)
        exp_risk    = st.slider("Expiry Risk (units expiring soon)", 0, 70, 10)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if st.button("🔮 Predict Next-Week Demand", key="btn_predict"):
        inputs = {
            "medicine_id": med_id, "medicine_name": med_name,
            "category": category,  "medicine_importance": med_imp,
            "year": year, "week_number": week_num, "season": season,
            "current_stock": curr_stock, "previous_week_sales": prev_sales,
            "sales_last_2_week": sales_2w, "sales_last_4_week_average": float(sales_4avg),
            "flu_cases": flu, "dengue_cases": dengue, "covid_cases": covid,
            "supplier_delivery_days": sup_days, "stock_received": stock_recv,
            "expiry_risk": exp_risk, "is_holiday_week": int(is_hol),
        }

        try:
            X_inf    = build_inference_row(inputs)
            pred     = float(model.predict(X_inf)[0])
            pred     = max(0, round(pred))

            avg_daily = sales_4avg / 7
            risk      = classify_stockout_risk(
                pred, curr_stock, sup_days, avg_daily,
                low_thresh=low_thresh, critical_thresh=critical_thresh,
            )
            # apply reorder buffer
            rec_order = max(0, round(risk["recommended_order"] * (1 + reorder_buffer / 100)))

            # ── results layout ────────────────────────────────────────────────
            r1, r2, r3 = st.columns(3)

            with r1:
                # Gauge chart
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,
                    delta={"reference": sales_4avg, "valueformat": ".0f",
                           "font": {"size": 13}},
                    number={"font": {"size": 42, "color": "#00BFA5"}, "suffix": " units"},
                    title={"text": "Predicted Demand", "font": {"size": 13, "color": "#8B949E"}},
                    gauge={
                        "axis": {"range": [0, max(pred * 2.5, 100)],
                                 "tickcolor": "#30363D", "tickfont": {"size": 10}},
                        "bar":  {"color": "#00BFA5"},
                        "bgcolor": "#21262D",
                        "bordercolor": "#30363D",
                        "steps": [
                            {"range": [0,           pred * 0.75], "color": "#1C2128"},
                            {"range": [pred * 0.75, pred * 1.25], "color": "#21262D"},
                        ],
                        "threshold": {
                            "line":  {"color": "#FF9800", "width": 3},
                            "thickness": 0.8,
                            "value": sales_4avg,
                        },
                    }
                ))
                apply_theme(gauge, height=230, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})

            with r2:
                # Risk banner
                bc = risk["color"]
                st.markdown(
                    f"<div class='risk-banner' style='background:{bc}20;border:1.5px solid {bc};color:{bc}'>"
                    f"{risk['icon']} Stockout Risk: {risk['level']}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                for label, val in [
                    ("Weeks of Stock",    f"{risk['weeks_of_stock']} wks"),
                    ("Safety Stock",      f"{risk['safety_stock']} units"),
                    ("Recommended Order", f"<b style='color:#00BFA5'>{rec_order} units</b>"),
                    ("Lead Time",         f"{sup_days} days"),
                ]:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;margin:.3rem 0'>"
                        f"<span class='muted'>{label}</span><span>{val}</span></div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            with r3:
                # Importance badge + context
                imp_col = IMPORTANCE_COLORS.get(med_imp, "#8B949E")
                avg_color = "#F44336" if pred > sales_4avg * 1.15 else "#4CAF50"
                avg_trend = "▲" if pred >= sales_4avg else "▼"
                st.markdown(
                    f"<div class='card'>"
                    f"<div class='muted'>Medicine</div>"
                    f"<div style='font-size:1.05rem;font-weight:700;margin:.25rem 0'>{med_name}</div>"
                    f"<span class='align-badge' style='background:{imp_col}22;color:{imp_col};border:1px solid {imp_col}'>{med_imp}</span>"
                    f"<hr class='divider' style='margin:.6rem 0'>"
                    f"<div style='font-size:.8rem;color:#8B949E;margin-bottom:.3rem'>DEMAND CONTEXT</div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.85rem'>"
                    f"<span>Forecast</span><span style='color:#00BFA5;font-weight:700'>{pred}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.85rem'>"
                    f"<span>4-wk avg</span><span>{sales_4avg:.0f}</span></div>"
                    f"<div style='display:flex;justify-content:space-between;font-size:.85rem'>"
                    f"<span>vs avg</span>"
                    f"<span style='color:{avg_color}'>"
                    f"{avg_trend} {abs(pred - sales_4avg):.0f}</span></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Batch Demand Forecasting")
    st.markdown(
        "<span class='muted'>Upload a CSV with the same columns as the training data. "
        "The model will predict next-week demand for each row.</span>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        # template download
        template_cols = [
            "medicine_id", "medicine_name", "category", "year", "week_number",
            "season", "current_stock", "previous_week_sales", "sales_last_2_week",
            "sales_last_4_week_average", "flu_cases", "dengue_cases", "covid_cases",
            "supplier_delivery_days", "stock_received", "expiry_risk",
            "is_holiday_week", "medicine_importance",
        ]
        template_df = pd.DataFrame(columns=template_cols)
        csv_buf = io.StringIO()
        template_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇ Download Template",
            data=csv_buf.getvalue(),
            file_name="batch_template.csv",
            mime="text/csv",
        )

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            st.markdown(f"<span class='muted'>Loaded <b>{len(batch_df)}</b> rows</span>",
                        unsafe_allow_html=True)

            with st.spinner("Running feature engineering & predictions…"):
                model = load_model()

                # feature engineering on batch (needs next_week_demand placeholder)
                if "next_week_demand" not in batch_df.columns:
                    batch_df["next_week_demand"] = batch_df.get("previous_week_sales", 50)

                eng = engineer_features(batch_df)
                eng_clean = eng.dropna(subset=["demand_lag1"])

                preds = model.predict(eng_clean[FEATURES])
                preds = np.maximum(preds, 0).round().astype(int)

                out = eng_clean.copy()
                out["predicted_demand"] = preds
                out["weeks_of_stock"]   = (
                    out["current_stock"] / out["predicted_demand"].clip(lower=1)
                ).round(1)
                out["recommended_order"] = np.maximum(
                    0,
                    np.round(
                        out["predicted_demand"] * (1 + out["supplier_delivery_days"] / 7)
                        - out["current_stock"]
                    ).astype(int),
                )
                out["recommended_order"] = (out["recommended_order"] * (1 + reorder_buffer / 100)).round().astype(int)

                def _risk_label(row):
                    wks = row["weeks_of_stock"]
                    if wks <= critical_thresh: return "CRITICAL"
                    if wks <= low_thresh:      return "LOW"
                    if wks <= 2.5:             return "MODERATE"
                    return "HEALTHY"

                out["stockout_risk"] = out.apply(_risk_label, axis=1)

            # summary metrics
            m1, m2, m3, m4 = st.columns(4)
            for col, label, val, sub in [
                (m1, "Total Medicines", len(out["medicine_id"].unique()), "unique"),
                (m2, "Critical Risk",   (out["stockout_risk"] == "CRITICAL").sum(), "rows"),
                (m3, "Avg Forecast",    f"{out['predicted_demand'].mean():.1f}", "units/row"),
                (m4, "Need Reorder",    (out["recommended_order"] > 0).sum(), "rows"),
            ]:
                col.markdown(
                    f"<div class='metric-pill'><div class='label'>{label}</div>"
                    f"<div class='value'>{val}</div>"
                    f"<div class='sub'>{sub}</div></div>",
                    unsafe_allow_html=True,
                )

            # results table
            display_cols = [
                "medicine_id", "medicine_name", "medicine_importance",
                "current_stock", "predicted_demand",
                "weeks_of_stock", "stockout_risk", "recommended_order",
            ]
            avail = [c for c in display_cols if c in out.columns]
            st.dataframe(
                out[avail].sort_values("weeks_of_stock"),
                use_container_width=True,
                hide_index=True,
            )

            # download full results
            result_csv = out.to_csv(index=False)
            st.download_button(
                "⬇ Download Full Results CSV",
                data=result_csv,
                file_name="batch_forecast_results.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Batch processing error: {e}")
            import traceback; st.code(traceback.format_exc())
    else:
        st.markdown(
            "<div class='card' style='text-align:center;padding:2rem;color:#30363D'>"
            "<div style='font-size:2.5rem'>📋</div>"
            "<div style='margin-top:.5rem;color:#8B949E'>Upload a CSV to start batch forecasting</div>"
            "<div class='muted' style='margin-top:.3rem'>or download the template to build your input file</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown("### Portfolio Analytics Dashboard")

    try:
        model    = load_model()
        df_raw   = load_dataset()
        df_eng   = engineer_features(df_raw)
        df_eng   = df_eng.dropna(subset=["demand_lag1"]).reset_index(drop=True)
        preds_all = model.predict(df_eng[FEATURES])
        df_eng["predicted_demand"] = np.maximum(preds_all, 0)
        df_eng["weeks_of_stock"]   = (
            df_eng["current_stock"] / df_eng["predicted_demand"].clip(lower=1)
        ).round(2)

        def _risk(row):
            w = row["weeks_of_stock"]
            if w <= critical_thresh: return "CRITICAL"
            if w <= low_thresh:      return "LOW"
            if w <= 2.5:             return "MODERATE"
            return "HEALTHY"

        df_eng["stockout_risk"] = df_eng.apply(_risk, axis=1)

        # ── top KPIs ─────────────────────────────────────────────────────────
        latest = df_eng.sort_values("time_idx").groupby("medicine_id").last().reset_index()
        n_crit  = (latest["stockout_risk"] == "CRITICAL").sum()
        n_low   = (latest["stockout_risk"] == "LOW").sum()
        avg_cov = latest["weeks_of_stock"].median()
        n_med   = latest["medicine_id"].nunique()

        k1, k2, k3, k4 = st.columns(4)
        for col, label, val, sub, clr in [
            (k1, "Medicines Tracked",  n_med,           "in catalog",          "#E6EDF3"),
            (k2, "Critical Stockout",  n_crit,          "need immediate action","#F44336"),
            (k3, "Low Stock Alert",    n_low,           "reorder soon",         "#FF9800"),
            (k4, "Median Coverage",    f"{avg_cov:.1f} wks", "across portfolio", "#00BFA5"),
        ]:
            col.markdown(
                f"<div class='metric-pill'><div class='label'>{label}</div>"
                f"<div class='value' style='color:{clr}'>{val}</div>"
                f"<div class='sub'>{sub}</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── row 1: donut + bar ────────────────────────────────────────────────
        ch1, ch2 = st.columns([1, 2])

        with ch1:
            risk_counts = latest["stockout_risk"].value_counts().reindex(
                ["CRITICAL", "LOW", "MODERATE", "HEALTHY"], fill_value=0
            )
            donut = go.Figure(go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.62,
                marker_colors=["#F44336", "#FF9800", "#FDD835", "#4CAF50"],
                textinfo="label+percent",
                textfont=dict(size=11),
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            donut.add_annotation(
                text=f"<b>{len(latest)}</b><br><span style='font-size:10px'>medicines</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#E6EDF3"),
            )
            apply_theme(donut, height=260, title="Stockout Risk Distribution",
                        showlegend=False, margin=dict(l=5, r=5, t=35, b=5))
            st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

        with ch2:
            cat_stats = (
                df_eng.groupby("category")["predicted_demand"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("mean", ascending=True)
            )
            bar = go.Figure()
            bar.add_trace(go.Bar(
                y=cat_stats["category"],
                x=cat_stats["mean"],
                orientation="h",
                marker=dict(
                    color=cat_stats["mean"],
                    colorscale=[[0, "#164E50"], [1, "#00BFA5"]],
                    showscale=False,
                ),
                error_x=dict(type="data", array=cat_stats["std"].fillna(0),
                             color="#30363D", thickness=1.5, width=4),
                hovertemplate="<b>%{y}</b><br>Avg Demand: %{x:.1f}<extra></extra>",
            ))
            apply_theme(bar, height=260, title="Avg Predicted Demand by Category",
                        xaxis_title="Units / Week",
                        margin=dict(l=10, r=10, t=35, b=30))
            bar.update_yaxes(tickfont=dict(size=11))
            st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        # ── row 2: scatter + demand trend ────────────────────────────────────
        ch3, ch4 = st.columns([3, 2])

        with ch3:
            scatter_df = latest.copy()
            scatter_df["importance_str"] = scatter_df["medicine_importance"].astype(str)
            scatter = go.Figure()
            for imp in IMPORTANCE_ORDER:
                sub = scatter_df[scatter_df["importance_str"] == imp]
                if sub.empty: continue
                scatter.add_trace(go.Scatter(
                    x=sub["current_stock"],
                    y=sub["predicted_demand"],
                    mode="markers",
                    name=imp,
                    marker=dict(
                        color=IMPORTANCE_COLORS.get(imp, "#8B949E"),
                        size=7,
                        opacity=0.75,
                        line=dict(width=0.5, color="#0D1117"),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Stock: %{x}<br>Forecast: %{y:.0f} units<extra></extra>"
                    ),
                    customdata=sub[["medicine_name"]].values,
                ))
            # diagonal reference line
            max_v = max(scatter_df["current_stock"].max(), scatter_df["predicted_demand"].max())
            scatter.add_trace(go.Scatter(
                x=[0, max_v], y=[0, max_v],
                mode="lines",
                name="1:1 line",
                line=dict(color="#30363D", dash="dot", width=1),
                hoverinfo="skip",
            ))
            apply_theme(scatter, height=300,
                        title="Current Stock vs Predicted Demand (latest snapshot)",
                        xaxis_title="Current Stock (units)",
                        yaxis_title="Predicted Demand (units/wk)")
            st.plotly_chart(scatter, use_container_width=True, config={"displayModeBar": False})

        with ch4:
            # weekly demand trend for top 5 medicines by avg demand
            top5 = (
                df_eng.groupby("medicine_name")["predicted_demand"]
                .mean().nlargest(5).index.tolist()
            )
            trend_df = df_eng[df_eng["medicine_name"].isin(top5)].copy()
            trend_df["week_label"] = trend_df["year"].astype(str) + "-W" + trend_df["week_number"].astype(str).str.zfill(2)

            trend_fig = go.Figure()
            colors = ["#00BFA5", "#FF9800", "#2196F3", "#F44336", "#9C27B0"]
            for i, med in enumerate(top5):
                sub = trend_df[trend_df["medicine_name"] == med].sort_values("time_idx")
                # smooth with rolling
                smooth = sub["predicted_demand"].rolling(4, min_periods=1).mean()
                trend_fig.add_trace(go.Scatter(
                    x=sub["time_idx"],
                    y=smooth,
                    mode="lines",
                    name=med,
                    line=dict(color=colors[i % len(colors)], width=1.8),
                    hovertemplate=f"<b>{med}</b><br>Demand: %{{y:.0f}}<extra></extra>",
                ))
            apply_theme(trend_fig, height=300,
                        title="Demand Trend — Top 5 Medicines",
                        xaxis_title="Time Index",
                        yaxis_title="Smoothed Demand",
                        legend=dict(x=0, y=1, font=dict(size=10)),
                        margin=dict(l=10, r=10, t=35, b=30))
            st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})

        # ── priority reorder table ────────────────────────────────────────────
        st.markdown("#### 🚨 Priority Reorder Recommendations")
        reorder_tbl = latest.copy()
        reorder_tbl["recommended_order"] = np.maximum(
            0,
            np.round(
                reorder_tbl["predicted_demand"] * (1 + reorder_tbl["supplier_delivery_days"] / 7)
                - reorder_tbl["current_stock"]
            ).astype(int),
        )
        reorder_tbl["recommended_order"] = (
            reorder_tbl["recommended_order"] * (1 + reorder_buffer / 100)
        ).round().astype(int)

        priority_map = {"CRITICAL": 0, "LOW": 1, "MODERATE": 2, "HEALTHY": 3}
        imp_map      = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

        reorder_tbl["_risk_ord"] = reorder_tbl["stockout_risk"].map(priority_map)
        reorder_tbl["_imp_ord"]  = reorder_tbl["medicine_importance"].astype(str).map(imp_map)

        show_only_reorder = st.checkbox("Show only medicines needing reorder", value=True)
        if show_only_reorder:
            reorder_tbl = reorder_tbl[reorder_tbl["recommended_order"] > 0]

        tbl_cols = [
            "medicine_id", "medicine_name", "medicine_importance", "category",
            "current_stock", "predicted_demand", "weeks_of_stock",
            "stockout_risk", "recommended_order", "supplier_delivery_days",
        ]
        avail_cols = [c for c in tbl_cols if c in reorder_tbl.columns]
        display_tbl = (
            reorder_tbl[avail_cols + ["_risk_ord", "_imp_ord"]]
            .sort_values(["_risk_ord", "_imp_ord", "weeks_of_stock"])
            .drop(columns=["_risk_ord", "_imp_ord"])
            .reset_index(drop=True)
        )
        st.dataframe(display_tbl, use_container_width=True, hide_index=True)

        # export
        csv_out = display_tbl.to_csv(index=False)
        st.download_button(
            "⬇ Export Reorder List",
            data=csv_out,
            file_name="priority_reorder_list.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Dashboard error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_inspect:
    st.markdown("### Model Inspection & Feature Alignment")

    try:
        model = load_model()
        booster_features = model.get_booster().feature_names
        app_features     = FEATURES

        st.markdown("#### Feature Alignment Checker")
        st.markdown(
            "<span class='muted'>Verifies that the inference pipeline's feature list "
            "exactly matches what the trained model expects.</span>",
            unsafe_allow_html=True,
        )

        model_set = set(booster_features)
        app_set   = set(app_features)
        missing_from_app   = model_set - app_set
        extra_in_app       = app_set - model_set
        order_match        = list(booster_features) == list(app_features)
        all_match          = (not missing_from_app) and (not extra_in_app)

        fa1, fa2, fa3 = st.columns(3)
        for col, label, val, ok in [
            (fa1, "Model features",   len(booster_features), True),
            (fa2, "App features",     len(app_features),     True),
            (fa3, "Status",
             "✅ ALIGNED" if all_match else "❌ MISMATCH",
             all_match),
        ]:
            color = "#00BFA5" if ok and all_match else ("#F44336" if not ok else "#E6EDF3")
            col.markdown(
                f"<div class='metric-pill'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value' style='color:{color}'>{val}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if missing_from_app:
            st.error(f"❌ Missing from app pipeline: {sorted(missing_from_app)}")
        if extra_in_app:
            st.warning(f"⚠️ Extra features in app (not in model): {sorted(extra_in_app)}")
        if all_match and not order_match:
            st.warning("⚠️ Features match but ORDER differs — XGBoost requires correct order.")
        if all_match and order_match:
            st.success("✅ All features aligned in correct order. Inference pipeline is consistent with the trained model.")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        # ── feature importance ────────────────────────────────────────────────
        st.markdown("#### Feature Importance (Gain)")
        importance_dict = model.get_booster().get_score(importance_type="gain")
        imp_df = (
            pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
            .reset_index().rename(columns={"index": "feature"})
            .sort_values("gain", ascending=True)
            .tail(25)
        )

        imp_fig = go.Figure(go.Bar(
            x=imp_df["gain"],
            y=imp_df["feature"],
            orientation="h",
            marker=dict(
                color=imp_df["gain"],
                colorscale=[[0, "#164E50"], [0.5, "#00897B"], [1, "#00BFA5"]],
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Gain: %{x:.1f}<extra></extra>",
        ))
        apply_theme(imp_fig, height=520, title="Top 25 Features by Gain",
                    xaxis_title="Average Gain",
                    margin=dict(l=10, r=10, t=35, b=30))
        imp_fig.update_yaxes(tickfont=dict(size=10))
        st.plotly_chart(imp_fig, use_container_width=True, config={"displayModeBar": False})

        # ── model metadata ────────────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### Model Parameters")

        params = model.get_xgb_params()
        param_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        # ── full feature list ─────────────────────────────────────────────────
        with st.expander("Full feature list (model order)"):
            for i, f in enumerate(booster_features):
                in_app = f in app_set
                badge_color = "#00BFA5" if in_app else "#F44336"
                badge_label = "✓" if in_app else "✗"
                st.markdown(
                    f"<code style='color:{badge_color}'>[{badge_label}]</code> "
                    f"<span style='font-size:.85rem'>{i+1}. {f}</span>",
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Inspector error: {e}")
        import traceback; st.code(traceback.format_exc())
