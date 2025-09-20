import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

FEATURES = [
    "mean_speed",
    "harsh_brake_rate",
    "night_miles_pct",
    "urban_miles_pct",
    "weekend_miles_pct",
    "rush_hour_miles_pct",   
    "accel_var",
    "phone_use_pct",
    "total_miles",
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="models/risk_model.pkl")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    X = df[FEATURES]
    y = df["is_high_risk"]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("clf", clf)])
    cal = CalibratedClassifierCV(pipe, method="isotonic", cv=3)
    cal.fit(Xtr, ytr)

    proba = cal.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    print(f"Validation ROC-AUC: {auc:.3f}")

    joblib.dump({"model": cal, "features": FEATURES}, args.model)
    print(f"Saved model to {args.model}")
