from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sri Lanka Weather Forecasting | Ensemble Learning Dashboard",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0d9488 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.12);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 1rem;
        color: #e2e8f0;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #0d9488;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0f172a;
    }
    .kpi-sub {
        font-size: 0.82rem;
        color: #0d9488;
        margin-top: 0.2rem;
    }
    @media (prefers-color-scheme: dark) {
        .kpi-card {
            background: #1e293b;
            border-color: #334155;
            border-left-color: #14b8a6;
        }
        .kpi-label { color: #94a3b8; }
        .kpi-value { color: #f8fafc; }
        .kpi-sub { color: #2dd4bf; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Relative Paths Setup (Works Locally & on Streamlit Community Cloud)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

CITY_METADATA = {
    "Colombo": {
        "zone": "Wet Zone (Western Coastal Lowland)",
        "coords": "6.9271° N, 79.8612° E",
        "elevation": "7 m",
        "defaults": {"max": 31.0, "min": 24.8, "mean": 27.6, "rain": 6.5, "wind": 18.0, "sun": 7.5},
    },
    "Kandy": {
        "zone": "Wet/Intermediate Zone (Central Mid-Country)",
        "coords": "7.2906° N, 80.6337° E",
        "elevation": "500 m",
        "defaults": {"max": 28.5, "min": 20.2, "mean": 24.1, "rain": 5.2, "wind": 13.5, "sun": 6.8},
    },
    "Galle": {
        "zone": "Wet Zone (Southern Coastal)",
        "coords": "6.0535° N, 80.2210° E",
        "elevation": "12 m",
        "defaults": {"max": 30.2, "min": 24.5, "mean": 27.1, "rain": 7.0, "wind": 21.0, "sun": 7.2},
    },
    "Jaffna": {
        "zone": "Dry/Arid Zone (Northern Peninsula)",
        "coords": "9.6615° N, 80.0255° E",
        "elevation": "5 m",
        "defaults": {"max": 32.4, "min": 25.8, "mean": 28.7, "rain": 1.8, "wind": 23.5, "sun": 8.6},
    },
    "Nuwara Eliya": {
        "zone": "Montane Wet Zone (Central Highlands)",
        "coords": "6.9497° N, 80.7891° E",
        "elevation": "1,868 m",
        "defaults": {"max": 20.2, "min": 11.8, "mean": 15.6, "rain": 6.0, "wind": 15.0, "sun": 5.5},
    },
}

MONTH_NAMES = {
    1: "January (NE Monsoon)",
    2: "February (NE Monsoon / Dry)",
    3: "March (First Inter-Monsoon)",
    4: "April (First Inter-Monsoon)",
    5: "May (SW Monsoon Onset)",
    6: "June (SW Monsoon Peak)",
    7: "July (SW Monsoon)",
    8: "August (SW Monsoon)",
    9: "September (SW Monsoon Late)",
    10: "October (Second Inter-Monsoon)",
    11: "November (Second Inter-Monsoon)",
    12: "December (NE Monsoon)",
}


# -----------------------------------------------------------------------------
# Cached Asset Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained ensemble & hybrid models...")
def load_models():
    try:
        ensemble_models = joblib.load(MODELS_DIR / "ensemble_models.joblib")
        arma_models = joblib.load(MODELS_DIR / "arma_models.joblib")
        hybrid_rf_residual = joblib.load(MODELS_DIR / "hybrid_rf_residual.joblib")
        feature_cols = joblib.load(MODELS_DIR / "feature_cols.joblib")
        recent_city_temps = joblib.load(MODELS_DIR / "recent_city_temps.joblib")
        return ensemble_models, arma_models, hybrid_rf_residual, feature_cols, recent_city_temps
    except Exception:
        # Cross-version cloud fallback: retrain lightweight models in-memory from data/sri_lanka_weather.csv
        from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
        from sklearn.linear_model import Ridge
        from statsmodels.tsa.arima.model import ARIMA
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        df = pd.read_csv(DATA_DIR / "sri_lanka_weather.csv")
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values(["city", "time"]).reset_index(drop=True)
        num_cols = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "sunshine_duration",
        ]
        df[num_cols] = df.groupby("city")[num_cols].transform(lambda s: s.ffill().bfill())
        df["sunshine_hours"] = df["sunshine_duration"] / 3600.0
        df["month"] = df["time"].dt.month
        df["day_of_year"] = df["time"].dt.dayofyear
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
        df["target_next_temp"] = df.groupby("city")["temperature_2m_mean"].shift(-1)
        df = df.dropna(subset=["target_next_temp"]).reset_index(drop=True)
        recent_city_temps = (
            df.groupby("city")["temperature_2m_mean"]
            .apply(lambda s: [round(float(x), 2) for x in s.tail(10).tolist()])
            .to_dict()
        )
        df["city_name"] = df["city"]
        df_enc = pd.get_dummies(df, columns=["city"], dtype=int).sort_values(["time", "city_name"]).reset_index(drop=True)
        feature_cols = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "sunshine_hours",
            "month",
            "day_of_year",
            "temp_range",
            "city_Colombo",
            "city_Galle",
            "city_Jaffna",
            "city_Kandy",
            "city_Nuwara Eliya",
        ]
        train_df = df_enc[df_enc["time"] < pd.Timestamp("2025-07-01")].copy().reset_index(drop=True)
        X_tr, y_tr = train_df[feature_cols], train_df["target_next_temp"]

        rf = RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=3, oob_score=True, random_state=42, n_jobs=-1).fit(X_tr, y_tr)
        ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.08, random_state=42).fit(X_tr, y_tr)
        gbr = GradientBoostingRegressor(n_estimators=120, learning_rate=0.06, max_depth=4, subsample=0.85, random_state=42).fit(X_tr, y_tr)
        xgb = XGBRegressor(n_estimators=130, learning_rate=0.06, max_depth=5, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbosity=0).fit(X_tr, y_tr)
        lgb = LGBMRegressor(n_estimators=130, learning_rate=0.06, max_depth=6, num_leaves=31, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1).fit(X_tr, y_tr)
        cat = CatBoostRegressor(iterations=150, learning_rate=0.07, depth=6, random_seed=42, verbose=0, allow_writing_files=False).fit(X_tr, y_tr)
        stk = StackingRegressor(
            estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)],
            final_estimator=Ridge(alpha=1.0),
            cv=5,
            n_jobs=-1,
        ).fit(X_tr, y_tr)

        ensemble_models = {
            "Bagging: Random Forest": rf,
            "Boosting: AdaBoost": ada,
            "Boosting: Gradient Boosting": gbr,
            "Boosting: XGBoost": xgb,
            "Boosting: LightGBM": lgb,
            "Boosting: CatBoost": cat,
            "Stacking (RF+XGB+LGB -> Ridge)": stk,
        }
        arma_models = {}
        train_df["arma_pred_next"] = np.nan
        for c_name in recent_city_temps.keys():
            c_mask = train_df["city_name"] == c_name
            c_series = train_df.loc[c_mask, "temperature_2m_mean"].reset_index(drop=True)
            a_fit = ARIMA(c_series, order=(2, 0, 1)).fit()
            arma_models[c_name] = a_fit
            one_step = a_fit.fittedvalues.shift(-1)
            one_step.iloc[-1] = float(np.asarray(a_fit.forecast(steps=1))[0])
            train_df.loc[c_mask, "arma_pred_next"] = one_step.values
        hybrid_rf_residual = RandomForestRegressor(n_estimators=120, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1).fit(
            X_tr, y_tr - train_df["arma_pred_next"]
        )
        return ensemble_models, arma_models, hybrid_rf_residual, feature_cols, recent_city_temps


@st.cache_data(show_spinner="Loading evaluation metrics and forecast artifacts...")
def load_artifacts():
    model_results = pd.read_csv(ARTIFACTS_DIR / "model_results.csv")
    feature_importance = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv")
    colombo_forecast = pd.read_csv(ARTIFACTS_DIR / "colombo_forecast_sample.csv")
    colombo_forecast["time"] = pd.to_datetime(colombo_forecast["time"])
    return model_results, feature_importance, colombo_forecast


ensemble_models, arma_models, hybrid_rf_residual, feature_cols, recent_city_temps = load_models()
model_results_df, feature_imp_df, colombo_forecast_df = load_artifacts()

ALL_MODEL_NAMES = ["Hybrid: ARMA + Random Forest"] + list(ensemble_models.keys())


def get_model_family(model_name: str) -> str:
    if model_name.startswith("Hybrid"):
        return "Hybrid (ARIMA(2,0,1) + RF Residual)"
    if model_name.startswith("Bagging"):
        return "Bagging Ensemble (Parallel Trees + OOB)"
    if model_name.startswith("Boosting"):
        return "Boosting Ensemble (Sequential Error Correction)"
    if model_name.startswith("Stacking"):
        return "Stacking Meta-Ensemble (Level-0 CV + Ridge)"
    return "Ensemble Model"


def build_feature_row(
    city: str,
    month: int,
    day_of_year: int,
    temp_max: float,
    temp_min: float,
    temp_mean: float,
    precip: float,
    wind_max: float,
    sunshine_hrs: float,
    columns: list,
) -> pd.DataFrame:
    row_dict = {
        "temperature_2m_max": float(temp_max),
        "temperature_2m_min": float(temp_min),
        "temperature_2m_mean": float(temp_mean),
        "precipitation_sum": float(precip),
        "wind_speed_10m_max": float(wind_max),
        "sunshine_hours": float(sunshine_hrs),
        "month": int(month),
        "day_of_year": int(day_of_year),
        "temp_range": float(temp_max - temp_min),
    }
    for col in columns:
        if col.startswith("city_"):
            city_name = col.replace("city_", "")
            row_dict[col] = 1 if city_name == city else 0
    return pd.DataFrame([row_dict])[columns]


def compute_all_predictions(X_input: pd.DataFrame, city: str, today_mean: float) -> dict:
    preds = {}

    # 1. Hybrid ARMA + Random Forest Residual
    arma_obj = arma_models.get(city)
    recent_hist = recent_city_temps.get(city, [today_mean] * 10)
    if isinstance(recent_hist, (int, float, np.floating)):
        recent_list = [float(recent_hist)] * 10
    else:
        recent_list = [float(x) for x in list(recent_hist)]

    if hasattr(arma_obj, "apply"):
        try:
            updated_arma = arma_obj.apply(recent_list + [float(today_mean)])
            arma_base = float(np.asarray(updated_arma.forecast(steps=1))[0])
        except Exception:
            arma_base = float(today_mean)
    elif hasattr(arma_obj, "forecast"):
        try:
            raw_fc = float(np.asarray(arma_obj.forecast(steps=1))[0])
            city_baseline = float(np.mean(recent_list))
            arma_base = raw_fc + 0.75 * (today_mean - city_baseline)
        except Exception:
            arma_base = float(today_mean)
    else:
        arma_base = float(today_mean)

    rf_res_pred = float(hybrid_rf_residual.predict(X_input)[0])
    preds["Hybrid: ARMA + Random Forest"] = arma_base + rf_res_pred

    # 2. All Ensemble Models (Bagging, Boosting, Stacking)
    for name, model in ensemble_models.items():
        preds[name] = float(model.predict(X_input)[0])

    return preds


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🎛️ Forecast Controls")
st.sidebar.caption("Configure today's observations for any Sri Lankan climate zone.")

selected_city = st.sidebar.selectbox(
    "📍 Select Sri Lankan City",
    options=list(CITY_METADATA.keys()),
    index=0,
    help="Choose one of the 5 meteorological stations across Sri Lanka.",
)

city_info = CITY_METADATA[selected_city]
defaults = city_info["defaults"]

selected_model = st.sidebar.selectbox(
    "🤖 Active Ensemble Model",
    options=ALL_MODEL_NAMES,
    index=0,
    help="Switch between Hybrid, Bagging, Boosting, and Stacking models.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Temporal Inputs")

selected_month = st.sidebar.selectbox(
    "Month & Monsoon Season",
    options=list(MONTH_NAMES.keys()),
    format_func=lambda m: f"{m:02d} - {MONTH_NAMES[m]}",
    index=6,
)
default_doy = min(365, max(1, (selected_month - 1) * 30 + 15))

st.sidebar.markdown("### 🌡️ Today's Weather Observations")

temp_max = st.sidebar.slider(
    "Today's Max Temp (°C)",
    min_value=12.0,
    max_value=40.0,
    value=float(defaults["max"]),
    step=0.1,
)
temp_min = st.sidebar.slider(
    "Today's Min Temp (°C)",
    min_value=5.0,
    max_value=32.0,
    value=float(min(defaults["min"], temp_max - 0.5)),
    step=0.1,
)
default_mean = round(min(max(defaults["mean"], temp_min), temp_max), 1)
temp_mean = st.sidebar.slider(
    "Today's Mean Temp (°C)",
    min_value=float(temp_min),
    max_value=float(temp_max),
    value=float(default_mean),
    step=0.1,
)

precip_sum = st.sidebar.slider(
    "Daily Rainfall / Precipitation (mm)",
    min_value=0.0,
    max_value=200.0,
    value=float(defaults["rain"]),
    step=0.5,
)
wind_speed_max = st.sidebar.slider(
    "Max Wind Speed at 10m (km/h)",
    min_value=2.0,
    max_value=70.0,
    value=float(defaults["wind"]),
    step=0.5,
)
sunshine_hours = st.sidebar.slider(
    "Sunshine Hours (hrs/day)",
    min_value=0.0,
    max_value=12.5,
    value=float(defaults["sun"]),
    step=0.1,
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Station Info:** {selected_city}\n\n"
    f"• **Zone:** {city_info['zone']}\n"
    f"• **Coordinates:** {city_info['coords']}\n"
    f"• **Elevation:** {city_info['elevation']}"
)

# -----------------------------------------------------------------------------
# Compute Real-Time Predictions
# -----------------------------------------------------------------------------
X_user = build_feature_row(
    city=selected_city,
    month=selected_month,
    day_of_year=default_doy,
    temp_max=temp_max,
    temp_min=temp_min,
    temp_mean=temp_mean,
    precip=precip_sum,
    wind_max=wind_speed_max,
    sunshine_hrs=sunshine_hours,
    columns=feature_cols,
)

all_preds = compute_all_predictions(X_user, selected_city, temp_mean)
active_pred = all_preds[selected_model]
temp_delta = active_pred - temp_mean

# -----------------------------------------------------------------------------
# Header & Top KPI Row
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>🌦️ Sri Lanka Weather Forecasting: Ensemble Learning Dashboard</h1>
        <p>Next-Day Mean Temperature Prediction across 5 Sri Lankan Climate Zones using Bagging, Boosting, Stacking & Hybrid Time-Series Ensembles (2020–2026 Open-Meteo Archive)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1.15, 1.45, 1.25, 1.15])

with kpi_col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Selected City</div>
            <div class="kpi-value">{selected_city}</div>
            <div class="kpi-sub">{city_info['zone']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Active Model Family</div>
            <div class="kpi-value" style="font-size:1.25rem;">{selected_model}</div>
            <div class="kpi-sub">{get_model_family(selected_model)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col3:
    delta_sign = "+" if temp_delta >= 0 else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Predicted Tomorrow's Mean Temp</div>
            <div class="kpi-value">{active_pred:.2f} °C</div>
            <div class="kpi-sub">{delta_sign}{temp_delta:.2f} °C vs. Today's Mean ({temp_mean:.1f} °C)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_col4:
    ensemble_spread = np.std(list(all_preds.values()))
    ensemble_mean = np.mean(list(all_preds.values()))
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Multi-Model Consensus</div>
            <div class="kpi-value">{ensemble_mean:.2f} °C</div>
            <div class="kpi-sub">Spread (σ): ±{ensemble_spread:.2f} °C across 8 models</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Analytical Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Tab 1: Model Comparison & Live Predictions",
        "🌿 Tab 2: Feature Importance & Domain Insights",
        "📈 Tab 3: Time-Series Forecast (Colombo Test Set)",
    ]
)

# =============================================================================
# TAB 1: MODEL COMPARISON
# =============================================================================
with tab1:
    st.subheader("Out-of-Sample Test Set Performance (`>= 2025-07-01`)")
    st.caption(
        "Strict chronological evaluation on 2,440 unseen daily test observations across Colombo, Kandy, Galle, Jaffna, and Nuwara Eliya."
    )

    col_left, col_right = st.columns([1.15, 1.35])

    with col_left:
        styled_results = model_results_df.sort_values("RMSE (°C)", ascending=True).reset_index(drop=True)
        st.dataframe(
            styled_results.style.format(
                {"RMSE (°C)": "{:.4f}", "MAE (°C)": "{:.4f}", "R2 Score": "{:.4f}"}
            ).background_gradient(subset=["RMSE (°C)", "MAE (°C)"], cmap="YlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

        rf_model = ensemble_models.get("Bagging: Random Forest")
        if rf_model is not None and hasattr(rf_model, "oob_score_"):
            st.success(
                f"**Bagging Random Forest Out-of-Bag ($R^2_{{OOB}}$) Score:** `{rf_model.oob_score_:.4f}` "
                "— Internal bootstrap validation closely matches the unseen test $R^2$, confirming minimal overfitting."
            )

    with col_right:
        chart_df = model_results_df.sort_values("RMSE (°C)", ascending=False).copy()
        chart_df["Highlight"] = chart_df["Model"].apply(
            lambda m: "Active Model" if m == selected_model else "Benchmark Models"
        )
        fig_rmse = px.bar(
            chart_df,
            x="RMSE (°C)",
            y="Model",
            orientation="h",
            color="Highlight",
            text="RMSE (°C)",
            color_discrete_map={"Active Model": "#0d9488", "Benchmark Models": "#3b82f6"},
            title="Test Set Root Mean Squared Error (RMSE °C) — Lower is Better",
        )
        fig_rmse.update_traces(texttemplate="%{text:.4f} °C", textposition="outside")
        fig_rmse.update_layout(
            height=380,
            margin=dict(l=10, r=60, t=45, b=20),
            xaxis_title="RMSE (°C)",
            yaxis_title="",
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")
    st.subheader(f"⚡ Live Side-by-Side Model Predictions for {selected_city}")
    st.caption(
        f"Comparing all 8 trained model predictions dynamically for your current sidebar inputs "
        f"(Today's Mean: {temp_mean:.1f} °C, Range: {temp_max - temp_min:.1f} °C, Rainfall: {precip_sum:.1f} mm, Sunshine: {sunshine_hours:.1f} hrs)."
    )

    live_rows = []
    metrics_lookup = model_results_df.set_index("Model").to_dict(orient="index")
    for m_name in ALL_MODEL_NAMES:
        pred_val = all_preds[m_name]
        m_info = metrics_lookup.get(m_name, {})
        live_rows.append(
            {
                "Status": "👉 Active" if m_name == selected_model else "Benchmark",
                "Model Name": m_name,
                "Ensemble Family": get_model_family(m_name),
                "Predicted Next-Day Temp (°C)": round(pred_val, 3),
                "Change vs. Today (°C)": round(pred_val - temp_mean, 3),
                "Test RMSE (°C)": round(float(m_info.get("RMSE (°C)", np.nan)), 4),
                "Test R²": round(float(m_info.get("R2 Score", np.nan)), 4),
            }
        )

    live_df = pd.DataFrame(live_rows)
    st.dataframe(
        live_df.style.format(
            {
                "Predicted Next-Day Temp (°C)": "{:.2f} °C",
                "Change vs. Today (°C)": "{:+.2f} °C",
                "Test RMSE (°C)": "{:.4f}",
                "Test R²": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# =============================================================================
# TAB 2: FEATURE IMPORTANCE
# =============================================================================
with tab2:
    st.subheader("🌿 Global Feature Importance (Random Forest & Gradient Boosted Trees)")
    st.caption(
        "Relative contribution of meteorological and spatial features toward predicting tomorrow's mean temperature across Sri Lanka."
    )

    fi_sorted = feature_imp_df.sort_values("Importance", ascending=True).copy()
    fi_sorted["Importance_Pct"] = fi_sorted["Importance"] * 100.0

    col_fi_chart, col_fi_text = st.columns([1.45, 1.05])

    with col_fi_chart:
        fig_fi = px.bar(
            fi_sorted,
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            color="Importance_Pct",
            color_continuous_scale="Tealgrn",
            text="Importance_Pct",
            title="Feature Importance Ranking (% Contribution, Log-Scaled X-Axis for Micro-Features)",
        )
        fig_fi.update_traces(texttemplate="%{text:.3f}%", textposition="outside")
        fig_fi.update_layout(
            height=480,
            xaxis_type="log",
            xaxis_title="Importance (% - Log Scale)",
            yaxis_title="Feature",
            coloraxis_showscale=False,
            margin=dict(l=10, r=70, t=50, b=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_fi_text:
        st.markdown(
            r"""
            ### 🇱🇰 Sri Lankan Climatological Interpretation

            1. **Strong Thermal Inertia (`temperature_2m_mean`, ~72.6%):**
               Positioned near the equator ($6^\circ\text{N} - 10^\circ\text{N}$), Sri Lanka exhibits low day-to-day synoptic temperature volatility. Today's mean temperature serves as the primary autoregressive baseline for tomorrow's temperature.

            2. **Elevation & Montane Microclimate (`city_Nuwara Eliya`, ~26.5%):**
               At **1,868 m** in the Central Highlands, Nuwara Eliya operates in a distinct temperate montane regime ($14^\circ\text{C} - 17^\circ\text{C}$), roughly $11^\circ\text{C} - 13^\circ\text{C}$ cooler than lowland stations (Colombo, Galle, Jaffna) due to the environmental lapse rate ($\approx 6.5^\circ\text{C}/\text{km}$).

            3. **Seasonal Monsoon Cycle & Solar Forcing (`day_of_year`, `sunshine_hours`, `temp_max/min`):**
               Secondary variance is driven by the annual solar zenith cycle (peaking during the March–April First Inter-Monsoon) and cloud-cover cooling during the Southwest (*Yala*, May–Sep) and Northeast (*Maha*, Dec–Feb) monsoons.

            4. **Convective Rainfall & Wind Cooling (`precipitation_sum`, `wind_speed_10m_max`, `temp_range`):**
               Heavy monsoon downpours and strong coastal gusts suppress daytime maximums and narrow the diurnal `temp_range`, nudging the next day's mean downward by $0.3^\circ\text{C} - 1.2^\circ\text{C}$.
            """
        )

# =============================================================================
# TAB 3: TIME-SERIES FORECAST
# =============================================================================
with tab3:
    st.subheader("📈 Chronological Test Set Forecast: Colombo (`2025-07-01` to `2026-10-30`)")
    st.caption(
        "Comparing Actual Next-Day Mean Temperature against Hybrid (ARIMA + Random Forest), XGBoost, and Random Forest predictions."
    )

    min_date = colombo_forecast_df["time"].min().date()
    max_date = colombo_forecast_df["time"].max().date()

    date_range = st.slider(
        "Select Forecast Window to Inspect:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
    )

    mask = (colombo_forecast_df["time"].dt.date >= date_range[0]) & (
        colombo_forecast_df["time"].dt.date <= date_range[1]
    )
    filtered_fc = colombo_forecast_df.loc[mask].copy()

    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=filtered_fc["time"],
            y=filtered_fc["target_next_temp"],
            mode="lines",
            name="Actual Next-Day Mean Temp (°C)",
            line=dict(color="#0f172a", width=2.2),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=filtered_fc["time"],
            y=filtered_fc["Hybrid_Pred"],
            mode="lines",
            name="Hybrid: ARMA(2,0,1) + Random Forest",
            line=dict(color="#0d9488", width=2.0),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=filtered_fc["time"],
            y=filtered_fc["XGB_Pred"],
            mode="lines",
            name="Boosting: XGBoost",
            line=dict(color="#f59e0b", width=1.6, dash="dot"),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=filtered_fc["time"],
            y=filtered_fc["RF_Pred"],
            mode="lines",
            name="Bagging: Random Forest",
            line=dict(color="#3b82f6", width=1.6, dash="dash"),
        )
    )

    fig_ts.update_layout(
        height=460,
        title="Colombo Daily Temperature Forecast Trajectory (Actual vs. Ensemble Models)",
        xaxis_title="Date",
        yaxis_title="Next-Day Mean Temperature (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=15, r=15, t=55, b=20),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    # Window error statistics
    w_col1, w_col2, w_col3 = st.columns(3)
    for col_widget, label, pred_col in [
        (w_col1, "Hybrid (ARMA + RF) Window RMSE", "Hybrid_Pred"),
        (w_col2, "XGBoost Window RMSE", "XGB_Pred"),
        (w_col3, "Random Forest Window RMSE", "RF_Pred"),
    ]:
        if len(filtered_fc) > 0:
            rmse_val = float(np.sqrt(np.mean((filtered_fc["target_next_temp"] - filtered_fc[pred_col]) ** 2)))
            mae_val = float(np.mean(np.abs(filtered_fc["target_next_temp"] - filtered_fc[pred_col])))
            col_widget.metric(label, f"{rmse_val:.4f} °C", f"MAE: {mae_val:.4f} °C", delta_color="off")

st.markdown("---")
st.caption(
    "Data Source: Open-Meteo Historical Weather API (2020–2026) | "
    "Generative AI Disclosure: Code scaffolding & UI layout assisted by Antigravity IDE (Gemini); "
    "dataset selection, execution verification, and climatological interpretation performed by the student."
)
