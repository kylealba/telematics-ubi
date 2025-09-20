import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="UBI Dashboard", layout="wide")
st.title("Usage-Based Insurance Dashboard (POC)")

features_path = st.text_input("Path to driver_features.csv", "data/driver_features.csv")
model_path = st.text_input("Path to trained model (.pkl)", "models/risk_model.pkl")

col1, col2 = st.columns(2)

# Keep a scored dataframe available for the coaching tips + download section
df_scored = None

with col1:
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        st.subheader("Driver Feature Snapshot")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        df = None
        st.info("Generate features first to see data (see README).")

with col2:
    if os.path.exists(model_path) and os.path.exists(features_path):
        bundle = joblib.load(model_path)
        model = bundle["model"]
        feature_cols = bundle["features"]

        df = pd.read_csv(features_path)
        df["risk_score"] = model.predict_proba(df[feature_cols])[:, 1]
        df["price_factor"] = 1.0 + 0.8 * (df["risk_score"] - 0.5)
        df["price_factor"] = df["price_factor"].clip(lower=0.85, upper=1.25)

        # Nicely format numeric columns
        df["risk_score"] = df["risk_score"].round(3)
        df["price_factor"] = df["price_factor"].round(3)

        st.subheader("Scores & Pricing Preview")
        st.dataframe(df[["driver_id", "risk_score", "price_factor"]].head(20),
                     use_container_width=True)

        df_scored = df.copy()
    else:
        st.info("Train the model to preview scores & pricing.")

st.markdown("---")
st.markdown("**Notes:** This POC uses synthetic data and a basic GBM model with isotonic calibration.")

# ---------------------------
# Coaching Tips (styled)
# ---------------------------
st.subheader("Coaching Tips")

def tip(row):
    msgs = []
    if row.get("harsh_brake_rate", 0) > 0.08:
        msgs.append("⚠️ Frequent harsh braking. Leave more following distance.")
    if row.get("night_miles_pct", 0) > 0.35:
        msgs.append("🌙 Lots of night driving. Consider earlier trips when possible.")
    if row.get("phone_use_pct", 0) > 0.05:
        msgs.append("📱 Phone use detected. Use Do Not Disturb while driving.")
    if row.get("weekend_miles_pct", 0) > 0.30:
        msgs.append("🎉 High weekend miles. Avoid late-night high-traffic areas.")
    if row.get("rush_hour_miles_pct", 0) > 0.35:
        msgs.append("🚦 Heavy rush-hour driving. Shift departure times to avoid congestion.")
    return " • ".join(msgs) or "✅ Looking good!"

df_for_tips = df_scored
if df_for_tips is None and os.path.exists(features_path):
    df_for_tips = pd.read_csv(features_path)

def color_tips(val: str):
    if isinstance(val, str) and any(sym in val for sym in ["⚠️", "📱", "🌙"]):
        return "color: red; font-weight: 700;"
    return "color: green; font-weight: 700;"

def row_bg_by_risk(row):
    p = row.get("risk_score", None)
    if p is None:
        return [""] * len(row)
    if p >= 0.85:
        return ["background-color: rgba(255, 0, 0, 0.15)"] * len(row)
    if p <= 0.65:
        return ["background-color: rgba(0, 255, 0, 0.12)"] * len(row)
    return [""] * len(row)

if isinstance(df_for_tips, pd.DataFrame):
    df_display = df_for_tips.copy()
    if "tips" not in df_display.columns:
        df_display["tips"] = df_display.apply(tip, axis=1)

    cols_to_show = [
        "driver_id",
        "harsh_brake_rate",
        "night_miles_pct",
        "phone_use_pct",
        "weekend_miles_pct",
        "risk_score",
        "tips",
    ]
    cols_to_show = [c for c in cols_to_show if c in df_display.columns]

    st.subheader("Coaching Tips (Styled)")
    styled = df_display[cols_to_show].head(20).style
    if "risk_score" in cols_to_show:
        styled = styled.apply(row_bg_by_risk, axis=1)
    if "tips" in cols_to_show:
        styled = styled.applymap(color_tips, subset=["tips"])
    st.dataframe(styled, use_container_width=True)
else:
    st.info("Tips will appear after features are generated (and look better once the model is loaded).")

# ---------------------------
# Download button (scores + tips)
# ---------------------------
st.markdown("---")
if df_scored is not None:
    export_df = df_scored.copy()
    if "tips" not in export_df.columns:
        export_df["tips"] = export_df.apply(tip, axis=1)

    export_cols = [
        "driver_id", "risk_score", "price_factor",
        "harsh_brake_rate", "night_miles_pct", "phone_use_pct",
        "weekend_miles_pct", "tips"
    ]
    if "rush_hour_miles_pct" in export_df.columns:
        export_cols.insert(-1, "rush_hour_miles_pct")

    csv = export_df[export_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download driver scores (CSV)",
        data=csv,
        file_name=f"driver_scores_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
