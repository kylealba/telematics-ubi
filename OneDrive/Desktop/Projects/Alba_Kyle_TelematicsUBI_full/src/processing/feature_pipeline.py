import argparse, pandas as pd, numpy as np

def build_features(events_path: str) -> pd.DataFrame:
    df = pd.read_parquet(events_path)

    # derive weekend flag from timestamp
    dt = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)  # Sat/Sun = 5/6

    # group once
    grp = df.groupby("driver_id")

    # basic aggregates
    total_miles = grp["miles"].sum()
    mean_speed = grp["speed_kmh"].mean()
    harsh_brake_rate = grp["brake_flag"].mean()
    phone_use_pct = grp["phone_use_flag"].mean()
    accel_var = grp["long_accel"].var()

    # weighted mile shares (use flags as 0/1 weights)
    night_miles = (df["miles"] * df["night_flag"]).groupby(df["driver_id"]).sum()
    urban_miles = (df["miles"] * df["urban_flag"]).groupby(df["driver_id"]).sum()
    weekend_miles = (df["miles"] * df["is_weekend"]).groupby(df["driver_id"]).sum()

    denom = total_miles.replace(0, 1e-9)  # avoid divide-by-zero
    night_miles_pct = night_miles / denom
    urban_miles_pct = urban_miles / denom
    weekend_miles_pct = weekend_miles / denom

    features = pd.DataFrame({
        "driver_id": total_miles.index,
        "mean_speed": mean_speed.values,
        "harsh_brake_rate": harsh_brake_rate.values,
        "night_miles_pct": night_miles_pct.values,
        "urban_miles_pct": urban_miles_pct.values,
        "weekend_miles_pct": weekend_miles_pct.values,
        "accel_var": accel_var.values,
        "phone_use_pct": phone_use_pct.values,
        "total_miles": total_miles.values,
    })

    # synthetic training label correlated with risky behavior
    risk_logit = (
        0.02 * (features["mean_speed"] - 55)
        + 5.0 * features["harsh_brake_rate"]
        + 1.5 * features["night_miles_pct"]
        + 0.8 * features["weekend_miles_pct"]  # new: weekends slightly riskier
        + 1.2 * features["phone_use_pct"]
        + 0.5 * features["accel_var"].fillna(0)
    )
    probs = 1 / (1 + np.exp(-risk_logit))
    rng = np.random.default_rng(123)
    features["is_high_risk"] = (rng.random(len(probs)) < probs).astype(int)

        # after: dt = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["is_weekend"] = (dt.dt.weekday >= 5).astype(int)

    # NEW: rush hour flag (approx local morning/evening peaks 07–10 & 16–19)
    local_dt = dt.tz_convert("US/Eastern")  # choose a single tz for the POC
    hour = local_dt.dt.hour
    df["is_rush"] = (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))).astype(int)

    # ... existing groupby/aggregations ...

    rush_miles = (df["miles"] * df["is_rush"]).groupby(df["driver_id"]).sum()

    denom = total_miles.replace(0, 1e-9)
    rush_hour_miles_pct = rush_miles / denom

    features = pd.DataFrame({
        "driver_id": total_miles.index,
        "mean_speed": mean_speed.values,
        "harsh_brake_rate": harsh_brake_rate.values,
        "night_miles_pct": night_miles_pct.values,
        "urban_miles_pct": urban_miles_pct.values,
        "weekend_miles_pct": weekend_miles_pct.values,
        "rush_hour_miles_pct": rush_hour_miles_pct.values,   # <-- NEW
        "accel_var": accel_var.values,
        "phone_use_pct": phone_use_pct.values,
        "total_miles": total_miles.values,
    })

    # in the synthetic label logic, give rush hour some weight (small)
    risk_logit = (
        0.02 * (features["mean_speed"] - 55)
        + 5.0 * features["harsh_brake_rate"]
        + 1.5 * features["night_miles_pct"]
        + 0.8 * features["weekend_miles_pct"]
        + 0.6 * features["rush_hour_miles_pct"]               # <-- NEW
        + 1.2 * features["phone_use_pct"]
        + 0.5 * features["accel_var"].fillna(0)
    )

    return features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/driver_features.csv")
    args = ap.parse_args()

    feats = build_features(args.events)
    feats.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(feats):,} drivers.")
