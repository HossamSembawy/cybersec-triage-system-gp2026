"""
Report Figures
Regenerates the report's result figures from the saved models, the local
datasets, and the saved system evaluation run.

Run from the repository root: python -m scripts.make_report_figures
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.url_module.preprocessor import extract_url_features

# Paths
OUTPUT_DIR = Path("latex report/figures")
NETWORK_CSV = Path(
    "datasets/network_instrusion&phishing_dataset/"
    "Wednesday-workingHours.pcap_ISCX.csv"
)
URL_CSV = Path("datasets/malicious_url_dataset/malicious_phish.csv")
NETWORK_MODEL = Path("models/network_module/network_model.joblib")
URL_MODEL = Path("models/url_module/url_model.joblib")
EVALUATION_RUN = Path("triage reports/evaluation/20260921T141056280853Z")

# Split settings copied from the training notebooks so the test sets match.
TEST_RATIO = 0.20
RANDOM_SEED = 42
MEDIUM_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.70
NETWORK_BENIGN_LABEL = "BENIGN"
URL_LABEL_NAMES = ["benign", "defacement", "phishing", "malware"]

# Reported by the DistilBERT notebook; re-running inference needs a GPU.
EMAIL_CONFUSION = np.array([[5922, 18], [32, 6402]])

# Figure size matches the report text width (A4 with 2.5 cm margins).
FIGURE_WIDTH = 6.3

# Colors: the first three categorical slots pass colorblind checks on white.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BENIGN_STRIP = "#e1e0d9"
SEQUENTIAL_BLUE = ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#104281"]
SEVERITY_COLORS = {"LOW": "#cde2fb", "MEDIUM": "#5598e7", "HIGH": "#184f95"}


def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": SECONDARY_INK,
        "axes.titlecolor": PRIMARY_INK,
        "xtick.color": MUTED_INK,
        "ytick.color": MUTED_INK,
        "xtick.labelcolor": SECONDARY_INK,
        "ytick.labelcolor": SECONDARY_INK,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def style_axis(axis):
    # Keep grid and frame recessive so the data carries the figure.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def save_figure(figure, filename):
    figure.savefig(OUTPUT_DIR / filename)
    plt.close(figure)
    print(f"Saved {OUTPUT_DIR / filename}")


def normalize_scores(raw_scores, score_lo, score_hi):
    return np.clip((raw_scores - score_lo) / (score_hi - score_lo), 0.0, 1.0)


def load_network_test_set():
    """Rebuild the notebook's held-out network split and score it."""
    network_data = pd.read_csv(NETWORK_CSV)
    network_data.columns = network_data.columns.str.strip()
    network_data = network_data.dropna(how="all")
    network_data = network_data.dropna(subset=["Label"])

    bundle = joblib.load(NETWORK_MODEL)
    features = network_data[bundle["feature_names"]]
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    attack_labels = (
        network_data["Label"] != NETWORK_BENIGN_LABEL
    ).astype(int).to_numpy()

    _, test_features, _, test_labels = train_test_split(
        features.to_numpy(dtype=float),
        attack_labels,
        test_size=TEST_RATIO,
        stratify=attack_labels,
        random_state=RANDOM_SEED,
    )
    raw_scores = -bundle["model"].score_samples(test_features)
    anomaly_scores = normalize_scores(
        raw_scores, bundle["score_lo"], bundle["score_hi"]
    )
    return raw_scores, anomaly_scores, test_labels


def load_url_test_predictions():
    """Rebuild the notebook's held-out URL split and predict it."""
    url_data = pd.read_csv(URL_CSV)
    url_data = url_data.dropna(subset=["url", "type"])
    url_data = url_data[url_data["type"].isin(URL_LABEL_NAMES)]

    feature_matrix = np.array([
        extract_url_features(url) for url in url_data["url"]
    ])
    labels = url_data["type"].map(
        {name: index for index, name in enumerate(URL_LABEL_NAMES)}
    ).to_numpy()

    _, test_features, _, test_labels = train_test_split(
        feature_matrix,
        labels,
        test_size=TEST_RATIO,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    bundle = joblib.load(URL_MODEL)
    predicted_labels = bundle["model"].predict(test_features)
    return test_labels, predicted_labels


def plot_network_roc(raw_scores, anomaly_scores, test_labels):
    false_positive_rate, true_positive_rate, _ = roc_curve(
        test_labels, raw_scores
    )
    area = roc_auc_score(test_labels, raw_scores)

    # Operating point used by the API: normalized score at the MEDIUM threshold.
    flagged = anomaly_scores >= MEDIUM_THRESHOLD
    operating_tpr = flagged[test_labels == 1].mean()
    operating_fpr = flagged[test_labels == 0].mean()

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH * 0.6, FIGURE_WIDTH * 0.52))
    axis.plot([0, 1], [0, 1], color=MUTED_INK, linewidth=1, linestyle="--")
    axis.plot(false_positive_rate, true_positive_rate, color=BLUE, linewidth=2)
    axis.scatter(
        [operating_fpr], [operating_tpr], s=45, color=ORANGE,
        edgecolor="white", linewidth=1.5, zorder=3,
    )
    axis.annotate(
        f"0.40 threshold\nTPR {operating_tpr:.3f}, FPR {operating_fpr:.3f}",
        xy=(operating_fpr, operating_tpr), xytext=(0.30, 0.42),
        color=SECONDARY_INK, fontsize=8,
        arrowprops={"arrowstyle": "-", "color": MUTED_INK, "linewidth": 0.8},
    )
    axis.text(0.55, 0.12, f"ROC-AUC {area:.4f}", color=PRIMARY_INK, fontsize=9)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    style_axis(axis)
    save_figure(figure, "network_roc.pdf")
    return area


def plot_network_score_distribution(anomaly_scores, test_labels):
    bins = np.linspace(0, 1, 26)
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.38))

    for label_value, name, color in [
        (0, "Benign flows", BLUE),
        (1, "Attack flows", ORANGE),
    ]:
        class_scores = anomaly_scores[test_labels == label_value]
        weights = np.full(len(class_scores), 1 / len(class_scores))
        zero_share = (class_scores == 0).mean()
        axis.hist(
            class_scores, bins=bins, weights=weights, histtype="step",
            linewidth=2, color=color,
            label=f"{name} ({zero_share:.0%} at 0)",
        )

    for threshold, name in [(MEDIUM_THRESHOLD, "MEDIUM"), (HIGH_THRESHOLD, "HIGH")]:
        axis.axvline(threshold, color=MUTED_INK, linewidth=1, linestyle="--")
        axis.text(threshold + 0.01, 0.55, name, color=SECONDARY_INK, fontsize=8)

    axis.set_yscale("log")
    axis.set_ylim(0.001, 1)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Normalized anomaly score")
    axis.set_ylabel("Share of class (log scale)")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    style_axis(axis)
    save_figure(figure, "network_score_distribution.pdf")


def plot_network_thresholds(anomaly_scores, test_labels):
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    precisions, recalls, f1_scores = [], [], []
    for threshold in thresholds:
        predicted = (anomaly_scores >= threshold).astype(int)
        precisions.append(precision_score(test_labels, predicted, zero_division=0))
        recalls.append(recall_score(test_labels, predicted, zero_division=0))
        f1_scores.append(f1_score(test_labels, predicted, zero_division=0))

    best_index = int(np.argmax(f1_scores))

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.4))
    for values, name, color in [
        (precisions, "Precision", BLUE),
        (recalls, "Recall", ORANGE),
        (f1_scores, "F1", AQUA),
    ]:
        axis.plot(thresholds, values, color=color, linewidth=2, label=name)
        # Direct labels, because the aqua line is below 3:1 contrast on white.
        axis.text(
            thresholds[-1] + 0.01, values[-1], name,
            color=SECONDARY_INK, fontsize=8, va="center",
        )

    for threshold, name in [(MEDIUM_THRESHOLD, "MEDIUM"), (HIGH_THRESHOLD, "HIGH")]:
        axis.axvline(threshold, color=MUTED_INK, linewidth=1, linestyle="--")
        axis.text(threshold + 0.01, 0.05, name, color=SECONDARY_INK, fontsize=8)

    # Mark the best F1 so the operating threshold can be compared against it.
    axis.scatter(
        [thresholds[best_index]], [f1_scores[best_index]], s=40, color=AQUA,
        edgecolor="white", linewidth=1.5, zorder=3,
    )
    axis.annotate(
        f"Best F1 {f1_scores[best_index]:.3f} at {thresholds[best_index]:.2f}",
        xy=(thresholds[best_index], f1_scores[best_index]),
        xytext=(0.47, 0.42), color=SECONDARY_INK, fontsize=8,
        arrowprops={"arrowstyle": "-", "color": MUTED_INK, "linewidth": 0.8},
    )

    axis.set_xlim(0.05, 1.02)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Threshold on normalized anomaly score")
    axis.set_ylabel("Attack-class score")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    style_axis(axis)
    save_figure(figure, "network_thresholds.pdf")
    return thresholds, precisions, recalls, f1_scores, best_index


def plot_confusion(axis, matrix, class_names, title):
    row_shares = matrix / matrix.sum(axis=1, keepdims=True)
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list(
        "sequential_blue", SEQUENTIAL_BLUE
    )
    axis.imshow(row_shares, cmap=color_map, vmin=0, vmax=1)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            share = row_shares[row, column]
            # Dark text on light cells and white text on dark cells.
            text_color = "white" if share > 0.55 else PRIMARY_INK
            axis.text(
                column, row, f"{matrix[row, column]:,}\n{share:.1%}",
                ha="center", va="center", fontsize=7.5, color=text_color,
            )

    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def plot_binary_confusions(network_matrix):
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.42))
    plot_confusion(
        axes[0], EMAIL_CONFUSION, ["Legitimate", "Phishing"],
        "Email (DistilBERT)",
    )
    plot_confusion(
        axes[1], network_matrix, ["Benign", "Attack"],
        "Network (Isolation Forest, 0.40)",
    )
    figure.tight_layout(w_pad=3)
    save_figure(figure, "binary_confusion.pdf")


def plot_url_confusion(url_matrix):
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH * 0.62, FIGURE_WIDTH * 0.56))
    plot_confusion(
        axis, url_matrix, ["Benign", "Defac.", "Phishing", "Malware"],
        "URL (Random Forest)",
    )
    save_figure(figure, "url_confusion.pdf")


def plot_system_decisions(summary):
    decisions = summary["paired_with_claude"]
    names = ["Email\nonly", "URL\nonly", "Network\nonly", "Highest\nseverity", "Claude\nreport"]
    keys = ["email", "url", "network", "max", "claude"]
    detected = [decisions[key]["tp"] for key in keys]
    malicious_total = decisions["max"]["tp"] + decisions["max"]["fn"]
    false_alarms = [decisions[key]["fp"] for key in keys]
    benign_total = decisions["max"]["fp"] + decisions["max"]["tn"]

    # Two panels rather than two y-axes, because the measures differ.
    figure, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.42))
    for axis, values, total, color, title in [
        (axes[0], detected, malicious_total, BLUE,
         f"Malicious cases flagged (of {malicious_total})"),
        (axes[1], false_alarms, benign_total, ORANGE,
         f"Benign cases flagged (of {benign_total})"),
    ]:
        bar_patches = axis.bar(range(len(names)), values, width=0.68, color=color)
        for bar_patch, value in zip(bar_patches, values):
            axis.text(
                bar_patch.get_x() + bar_patch.get_width() / 2, value + total * 0.02,
                str(value), ha="center", va="bottom",
                color=PRIMARY_INK, fontsize=8,
            )
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, fontsize=7.5)
        axis.set_ylim(0, total * 1.12)
        axis.set_title(title)
        style_axis(axis)
        axis.grid(axis="x", visible=False)
    figure.tight_layout(w_pad=2)
    save_figure(figure, "system_decisions.pdf")


def read_cases():
    with open(EVALUATION_RUN / "cases.csv", encoding="utf-8") as cases_file:
        cases = list(csv.DictReader(cases_file))
    # Group benign controls first so the escalation pattern is easy to read.
    return sorted(cases, key=lambda case: (case["is_malicious"] == "True", case["case_id"]))


def plot_case_grid(cases):
    rows = [
        ("Email", "email_severity", lambda case: case["email_source_label"] == "1"),
        ("URL", "url_severity", lambda case: case["url_source_label"] != "benign"),
        ("Network", "network_severity",
         lambda case: case["network_source_label"] != NETWORK_BENIGN_LABEL),
        ("Highest severity", "max_severity", None),
        ("Claude report", "claude_severity", None),
    ]
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.33))

    for column, case in enumerate(cases):
        # Top strip shows the case label that the decisions are scored against.
        strip_color = ORANGE if case["is_malicious"] == "True" else BENIGN_STRIP
        axis.add_patch(plt.Rectangle(
            (column, -1.1), 0.94, 0.55, color=strip_color, linewidth=0,
        ))
        for row_index, (_, field, is_malicious_source) in enumerate(rows):
            severity = case[field]
            top = row_index + (0.3 if row_index >= 3 else 0)
            axis.add_patch(plt.Rectangle(
                (column, top), 0.94, 0.9,
                color=SEVERITY_COLORS[severity], linewidth=0,
            ))
            text_color = PRIMARY_INK if severity == "LOW" else "white"
            axis.text(
                column + 0.47, top + 0.45, severity[0],
                ha="center", va="center", fontsize=6.5, color=text_color,
            )
            if is_malicious_source and is_malicious_source(case):
                axis.add_patch(plt.Rectangle(
                    (column + 0.04, top + 0.04), 0.86, 0.82,
                    fill=False, edgecolor=ORANGE, linewidth=1.6,
                ))

    labels = ["Case label"] + [name for name, _, _ in rows]
    positions = [-0.82] + [
        index + 0.45 + (0.3 if index >= 3 else 0) for index in range(len(rows))
    ]
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xticks([column + 0.47 for column in range(len(cases))])
    axis.set_xticklabels(
        [case["case_id"].split("-")[1] for case in cases], fontsize=6.5,
    )
    axis.set_xlim(-0.1, len(cases))
    axis.set_ylim(len(rows) + 0.35, -1.25)
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel("Case (benign controls first)")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=level_color)
        for level_color in SEVERITY_COLORS.values()
    ] + [
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
        plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=ORANGE, linewidth=1.6),
    ]
    legend_labels = [
        "LOW (L)", "MEDIUM (M)", "HIGH (H)",
        "Malicious case", "Malicious source sample",
    ]
    axis.legend(
        legend_handles, legend_labels, loc="upper center",
        bbox_to_anchor=(0.5, -0.32), ncol=5, fontsize=7.5,
        handlelength=1.2, columnspacing=1.2,
    )
    save_figure(figure, "case_grid.pdf")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_style()

    # Network figures, recomputed from the saved model and the held-out split
    print("Scoring the held-out network flows...")
    raw_scores, anomaly_scores, test_labels = load_network_test_set()
    area = plot_network_roc(raw_scores, anomaly_scores, test_labels)
    plot_network_score_distribution(anomaly_scores, test_labels)
    thresholds, precisions, recalls, f1_scores, best_index = plot_network_thresholds(
        anomaly_scores, test_labels
    )
    network_matrix = confusion_matrix(
        test_labels, (anomaly_scores >= MEDIUM_THRESHOLD).astype(int)
    )

    # URL figure, recomputed from the saved model and the held-out split
    print("Predicting the held-out URLs...")
    url_labels, url_predictions = load_url_test_predictions()
    url_matrix = confusion_matrix(url_labels, url_predictions)

    plot_binary_confusions(network_matrix)
    plot_url_confusion(url_matrix)

    # System evaluation figures, read from the saved 30-case run
    with open(EVALUATION_RUN / "summary.json", encoding="utf-8") as summary_file:
        summary = json.load(summary_file)
    plot_system_decisions(summary)
    plot_case_grid(read_cases())

    # Print the recomputed values so they can be checked against the report.
    print(f"\nNetwork ROC-AUC: {area:.4f}  test flows: {len(test_labels):,}")
    print(f"Network confusion matrix at 0.40:\n{network_matrix}")
    print(
        f"Best F1 threshold: {thresholds[best_index]:.2f} "
        f"(F1 {f1_scores[best_index]:.3f}, precision {precisions[best_index]:.3f}, "
        f"recall {recalls[best_index]:.3f})"
    )
    print(f"URL confusion matrix:\n{url_matrix}")


if __name__ == "__main__":
    main()
