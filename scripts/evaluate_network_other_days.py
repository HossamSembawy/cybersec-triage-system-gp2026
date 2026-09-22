"""
Network Cross-Day Evaluation
Scores the other CICIDS2017 days with the saved Wednesday Isolation Forest.
Nothing is retrained: the model and its benign score bounds are used as saved,
so each day is an out-of-sample test with attack types the model never saw.

Run from the repository root: python -m scripts.evaluate_network_other_days
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DATA_DIR = Path("datasets/network_instrusion&phishing_dataset")
NETWORK_MODEL = Path("models/network_module/network_model.joblib")
OTHER_DAYS = [
    ("Monday", "Monday-WorkingHours.pcap_ISCX.csv"),
    ("Tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("Thursday AM", "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("Thursday PM", "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("Friday AM", "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("Friday PM (PortScan)", "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("Friday PM (DDoS)", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
]
BENIGN_LABEL = "BENIGN"
MEDIUM_THRESHOLD = 0.40


def load_day(csv_path, feature_names):
    # Some CICIDS2017 label strings contain non-UTF-8 bytes.
    data = pd.read_csv(csv_path, encoding="latin-1", low_memory=False)
    data.columns = data.columns.str.strip()
    data = data.dropna(subset=["Label"])
    features = data[feature_names].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # The web-attack labels use a non-ASCII dash; normalize it for printing.
    labels = data["Label"].str.strip().str.replace(r"[^\x00-\x7f]+", "-", regex=True)
    return features.to_numpy(dtype=float), labels.to_numpy()


def score_day(bundle, features):
    raw_scores = -bundle["model"].score_samples(features)
    span = bundle["score_hi"] - bundle["score_lo"]
    anomaly_scores = np.clip((raw_scores - bundle["score_lo"]) / span, 0.0, 1.0)
    return raw_scores, anomaly_scores


def main():
    bundle = joblib.load(NETWORK_MODEL)
    rows = []

    for day, filename in OTHER_DAYS:
        features, labels = load_day(DATA_DIR / filename, bundle["feature_names"])
        raw_scores, anomaly_scores = score_day(bundle, features)
        is_attack = labels != BENIGN_LABEL
        flagged = anomaly_scores >= MEDIUM_THRESHOLD

        benign_count = int((~is_attack).sum())
        attack_count = int(is_attack.sum())
        false_positive_rate = flagged[~is_attack].mean() if benign_count else np.nan
        recall = flagged[is_attack].mean() if attack_count else np.nan
        # ROC-AUC uses the unclipped scores so clipping does not create ties.
        area = roc_auc_score(is_attack, raw_scores) if attack_count and benign_count else np.nan

        print(f"\n{day}: {benign_count:,} benign, {attack_count:,} attack")
        print(f"  FPR at 0.40: {false_positive_rate:.3f}   recall: {recall:.3f}   ROC-AUC: {area:.4f}")
        for attack in sorted(set(labels[is_attack])):
            mask = labels == attack
            print(f"    {attack:<32} n={int(mask.sum()):>7,}  recall={flagged[mask].mean():.3f}")
            rows.append({
                "day": day, "attack": attack, "flows": int(mask.sum()),
                "recall": round(float(flagged[mask].mean()), 3),
            })
        rows.append({
            "day": day, "attack": "ALL", "flows": attack_count,
            "benign_flows": benign_count,
            "benign_fpr": None if np.isnan(false_positive_rate) else round(float(false_positive_rate), 3),
            "recall": None if np.isnan(recall) else round(float(recall), 3),
            "roc_auc": None if np.isnan(area) else round(float(area), 4),
        })

    summary = pd.DataFrame(rows)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
