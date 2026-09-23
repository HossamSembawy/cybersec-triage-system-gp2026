# Explainable AI Cybersecurity Triage System

Multi-model threat correlation and decision support. CM3070 Final Year Project,
University of London BSc Computer Science, built on the CM3020 template
*Orchestrating AI Models to Achieve a Goal*.

The system runs three specialist detectors and combines their outputs into one
triage report:

- **Email:** a fine-tuned DistilBERT phishing classifier
- **URL:** a Random Forest on 15 lexical features (benign, defacement, phishing, malware)
- **Network:** an Isolation Forest anomaly detector trained on benign CICIDS2017 flows

An orchestration service sends the structured detector results (never the raw
email, URL, or flow) to the Claude API, which writes the combined severity,
evidence summary, and recommended analyst check. The language model does not
perform detection, and the API rejects a report whose severity is lower than the
strongest detector result.

**Research question:** can orchestrating specialized AI models into a unified
pipeline improve cybersecurity triage compared with isolated single-model
detection?

## Results

| Component | Held-out test set | Main result |
|---|---|---|
| Email (DistilBERT) | 12,374 emails | 99.60% accuracy, 99.50% phishing recall |
| URL (Random Forest) | 130,239 URLs | 0.9213 macro-F1, 94% accuracy |
| Network (Isolation Forest) | 138,541 flows | 0.8265 ROC-AUC, 0.674 attack recall at 0.40 |

On 30 synthetic cases assembled from held-out samples, taking the highest
detector severity flagged all 21 malicious cases, compared with 12 for the best
single detector, but also flagged 3 of 9 benign cases. The Claude report matched
that severity in every case. The three datasets share no hosts or timestamps, so
these cases test detection coverage, not correlation of real attacks. The full
analysis is in the project report.

## Repository layout

```
src/
  email_module/     DistilBERT preprocessing and prediction
  url_module/       lexical feature extraction and Random Forest prediction
  network_module/   flow feature ordering and Isolation Forest prediction
  orchestration/    Claude request, response schema, and validation
  api/              FastAPI app, cached model loaders, and routes
frontend/           React/Vite analyst dashboard
notebooks/          Colab training and evaluation notebooks, one per detector
scripts/            system evaluation, triage case runner, and report audit
tests/              pytest suite
```

## Setup

Trained models and datasets are **not** included in this repository because of
their size. Each detector is trained by running its notebook in Google Colab:

| Detector | Notebook | Dataset |
|---|---|---|
| Email | `notebooks/02_distilbert_finetune.ipynb` | [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| URL | `notebooks/03_url_randomforest.ipynb` | [Malicious URLs Dataset](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) |
| Network | `notebooks/04_network_isolationforest.ipynb` | [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), Wednesday file |

Place the trained files in:

```
models/email_module/     model.safetensors, config.json, tokenizer files
models/url_module/       url_model.joblib
models/network_module/   network_model.joblib
```

Then install the Python dependencies (Python 3.13):

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash; use venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

`scikit-learn` is pinned to 1.6.1 to match the Colab training environment. A
model saved with a different version can load but predict incorrectly.

## Running

**API:**

```bash
export ANTHROPIC_API_KEY=your-key   # only needed for the orchestration endpoint
uvicorn src.api.main:app --reload
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.

| Endpoint | Input |
|---|---|
| `POST /api/v1/email/analyze` | email text |
| `POST /api/v1/url/analyze` | a URL string |
| `POST /api/v1/network/analyze` | 78 flow features, as a named object or an ordered list |
| `POST /api/v1/orchestration/analyze` | one or more detector results |

**Dashboard** (with the API running):

```bash
cd frontend
npm install
npm run dev
```

The dashboard runs at `http://127.0.0.1:5173` and forwards API calls to port 8000.

## Tests

```bash
pytest
```

The tests use synthetic models and mocked Claude responses, so they run without
the trained model files or an API key. They check request validation, routing,
the shared severity thresholds, feature order, network value cleaning and score
normalization, and orchestration response validation. They verify software
behavior, not model accuracy.

## System evaluation

`scripts/evaluate_system.py` runs the 30-case comparison. It needs the running
API, the trained models, and held-out samples exported from the notebooks into
`triage reports/evaluation_inputs/`. Run `python -m scripts.evaluate_system --help`
for the options.

`python -m scripts.evaluate_network_other_days` applies the saved network model,
without retraining, to the other CICIDS2017 days and prints recall per attack
type and the benign false-positive rate for each file.

## Limitations

- The three datasets were collected independently, so the system cannot link
  an email, URL, and flow to one real incident.
- The email and URL detectors were tested only on a random split of a single
  dataset, not on external or time-separated data.
- On the other CICIDS2017 days, the network model flagged DDoS traffic but
  almost no brute-force, port-scan, web, or botnet flows. It detects
  high-volume denial-of-service traffic, not intrusions in general.
- The email test split was not audited for duplicate messages across splits.
- The URL model can misclassify bare popular domains such as `google.com` as
  phishing, because almost every benign training URL includes a path.
- The network score is a normalized anomaly score, not a probability of attack.
- Claude reports can still contain wording that implies a link between
  independent records. The report is a review aid for an analyst, not an
  automated response.
- Reports are not stored; there is no database.
