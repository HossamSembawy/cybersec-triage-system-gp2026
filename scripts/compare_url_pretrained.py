"""
Pre-trained URL Model Comparison
Compares the deployed Random Forest with a pre-trained Hugging Face URL
classifier (kmack/malicious-url-detection, a fine-tuned DistilBERT).

Many URLs in that model's training data also appear in our dataset, so both
models are scored only on held-out test URLs that it was never trained on.

Run from the repository root: python -m scripts.compare_url_pretrained
"""

from __future__ import annotations

import re
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.url_module.preprocessor import extract_url_features

URL_CSV = Path("datasets/malicious_url_dataset/malicious_phish.csv")
URL_MODEL = Path("models/url_module/url_model.joblib")
PRETRAINED_MODEL = "kmack/malicious-url-detection"
PRETRAINED_TRAIN_DATA = (
    "https://huggingface.co/datasets/kmack/Phishing_urls/resolve/main/data/"
    "train-00000-of-00001-d8afc95a165ea87b.parquet"
)
CACHE_DIR = Path("triage reports/pretrained_url")

# Split settings copied from the URL training notebook.
TEST_RATIO = 0.20
RANDOM_SEED = 42
LABEL_NAMES = ["benign", "defacement", "phishing", "malware"]
MEDIUM_THRESHOLD = 0.40
SAMPLE_SEED = 2026
SAMPLE_PER_CLASS = 1500
BATCH_SIZE = 64


def normalize(url):
    # The pre-trained model's data has no scheme or trailing slash.
    url = re.sub(r"^[a-z]+://", "", url.strip().lower())
    return url.rstrip("/")


def load_test_split():
    data = pd.read_csv(URL_CSV).dropna(subset=["url", "type"])
    data = data[data["type"].isin(LABEL_NAMES)]
    labels = data["type"].map({name: i for i, name in enumerate(LABEL_NAMES)}).to_numpy()
    _, test_positions = train_test_split(
        np.arange(len(data)), test_size=TEST_RATIO,
        stratify=labels, random_state=RANDOM_SEED,
    )
    return data.iloc[test_positions]


def load_pretrained_training_urls():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "kmack_train.parquet"
    if not path.exists():
        urllib.request.urlretrieve(PRETRAINED_TRAIN_DATA, path)
    return set(pd.read_parquet(path)["text"].map(normalize))


def score_random_forest(urls):
    bundle = joblib.load(URL_MODEL)
    # Timing includes feature extraction, the Random Forest's equivalent of tokenizing.
    start = time.perf_counter()
    features = np.array([extract_url_features(url) for url in urls])
    probabilities = bundle["model"].predict_proba(features)
    elapsed = time.perf_counter() - start
    benign_column = list(bundle["label_names"]).index("benign")
    return 1.0 - probabilities[:, benign_column], elapsed


def score_pretrained(urls):
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL).eval()
    malicious_column = [
        index for index, name in model.config.id2label.items() if name != "BENIGN"
    ][0]
    scores = []
    start = time.perf_counter()
    with torch.no_grad():
        for offset in range(0, len(urls), BATCH_SIZE):
            batch = [normalize(url) for url in urls[offset:offset + BATCH_SIZE]]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=128, return_tensors="pt")
            logits = model(**encoded).logits
            scores.extend(torch.softmax(logits, dim=-1)[:, malicious_column].tolist())
    return np.array(scores), time.perf_counter() - start


def summarize(name, scores, types, elapsed):
    flagged = scores >= MEDIUM_THRESHOLD
    malicious = types != "benign"
    true_positives = int((flagged & malicious).sum())
    precision = true_positives / max(int(flagged.sum()), 1)
    recall = true_positives / int(malicious.sum())
    row = {
        "model": name,
        "benign_fpr": round(float(flagged[~malicious].mean()), 3),
        "recall_all": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3),
    }
    for label in LABEL_NAMES[1:]:
        row[f"recall_{label}"] = round(float(flagged[types == label].mean()), 3)
    row["ms_per_url"] = round(1000 * elapsed / len(scores), 3)
    return row


def main():
    test = load_test_split()
    seen = test["url"].map(normalize).isin(load_pretrained_training_urls())
    print(f"Test URLs also in the pre-trained model's training data: {seen.mean():.1%}")
    print(test.assign(seen=seen).groupby("type")["seen"].mean().round(3).to_string())

    unseen = test[~seen]
    sample = pd.concat([
        rows.sample(n=min(SAMPLE_PER_CLASS, len(rows)), random_state=SAMPLE_SEED)
        for _, rows in unseen.groupby("type")
    ])
    urls = sample["url"].tolist()
    types = sample["type"].to_numpy()
    print(f"\nScoring {len(urls):,} unseen test URLs: {sample['type'].value_counts().to_dict()}")

    forest_scores, forest_time = score_random_forest(urls)
    pretrained_scores, pretrained_time = score_pretrained(urls)

    results = pd.DataFrame([
        summarize("Random Forest (deployed)", forest_scores, types, forest_time),
        summarize(f"{PRETRAINED_MODEL}", pretrained_scores, types, pretrained_time),
    ])
    print(f"\nThreshold {MEDIUM_THRESHOLD}; precision depends on this balanced sample.")
    print(results.T.to_string(header=False))


if __name__ == "__main__":
    main()
