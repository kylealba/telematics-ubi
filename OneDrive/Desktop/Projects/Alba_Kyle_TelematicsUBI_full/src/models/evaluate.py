import argparse, json, os, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def main(features_path: str, model_path: str, out_report: str, out_calib_png: str):
    # Load model bundle and data
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feats = bundle["features"]

    df = pd.read_csv(features_path)
    X = df[feats]
    y = df["is_high_risk"]

    # Predict probabilities
    p = model.predict_proba(X)[:, 1]

    # Metrics
    report = {
        "roc_auc": float(roc_auc_score(y, p)),
        "avg_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
    }

    # Calibration curve data (quantile bins are more stable)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    report["calibration"] = {"mean_pred": mean_pred.tolist(),
                             "frac_pos": frac_pos.tolist()}

    # Ensure docs directory exists
    os.makedirs(os.path.dirname(out_report) or ".", exist_ok=True)

    # Save JSON report
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_report}")

    # Plot & save calibration curve
    plt.figure(figsize=(5.5, 5))
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curve")
    plt.legend(loc="best")
    # Make sure output folder exists
    os.makedirs(os.path.dirname(out_calib_png) or ".", exist_ok=True)
    plt.savefig(out_calib_png, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_calib_png}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Path to driver_features.csv")
    ap.add_argument("--model", required=True, help="Path to models/risk_model.pkl")
    ap.add_argument("--out", default="docs/eval_report.json", help="JSON report path")
    ap.add_argument("--calib_png", default="docs/calibration.png", help="Calibration plot PNG path")
    args = ap.parse_args()

    main(args.features, args.model, args.out, args.calib_png)
