"""Evaluate exported samples and synthetic incidents through the local API."""

import argparse
import csv
import hashlib
import itertools
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_triage_cases import (
    DEFAULT_BASE_URL,
    DETECTOR_PATHS,
    ORCHESTRATION_PATH,
    PROJECT_ROOT,
    post_json,
)
from src.orchestration.schemas import OrchestrationRequest, OrchestrationResponse

MODULE_NAMES = ("email", "url", "network")
SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
CASE_SEED = 2026
OUTPUT_ROOT = PROJECT_ROOT / "triage reports" / "evaluation"


def save_json(path: Path, content: dict) -> None:
    # Replace the checkpoint only after the new JSON has been fully written.
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def create_manifest(input_dir: Path) -> dict:
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_seed": CASE_SEED,
        "positive_decision": "severity is MEDIUM or HIGH",
        "incident_truth": "at least one source sample is labeled malicious",
        "input_files": {},
        "samples": {},
        "cases": [],
    }
    sample_pools = {}
    generator = random.Random(CASE_SEED)

    for module_name in MODULE_NAMES:
        path = input_dir / f"{module_name}_evaluation_samples.json"
        content = path.read_bytes()
        export = json.loads(content)
        if export.get("module") != module_name:
            raise ValueError(f"Wrong module in {path.name}")
        samples = {}
        for sample in export["samples"]:
            sample_id = str(sample["source_row_index"])
            if sample_id in samples:
                raise ValueError(f"Repeated source row in {path.name}: {sample_id}")
            if type(sample["is_malicious"]) is not bool:
                raise ValueError(f"Expected a boolean is_malicious in {path.name}")
            label = sample["source_label"]
            if module_name == "email":
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("Email source labels must be 0 or 1")
                expected_malicious = label == 1
            elif module_name == "url":
                if label not in ("benign", "defacement", "phishing", "malware"):
                    raise ValueError("Unknown URL source label")
                expected_malicious = label != "benign"
            else:
                if not isinstance(label, str) or not label:
                    raise ValueError("Network source labels must be nonempty strings")
                expected_malicious = label != "BENIGN"
            if sample["is_malicious"] != expected_malicious:
                raise ValueError(f"Source label disagrees with is_malicious in {path.name}")
            request_key = {"email": "text", "url": "url", "network": "features"}[module_name]
            if set(sample["request"]) != {request_key}:
                raise ValueError(f"Unexpected request fields in {path.name}")
            samples[sample_id] = sample

        manifest["samples"][module_name] = samples
        manifest["input_files"][module_name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "metadata": {key: value for key, value in export.items() if key != "samples"},
        }
        sample_pools[module_name] = {}
        for is_malicious, required_count in ((False, 18), (True, 12)):
            sample_ids = sorted(
                sample_id for sample_id, sample in samples.items()
                if sample["is_malicious"] == is_malicious
            )
            if len(sample_ids) < required_count:
                raise ValueError(
                    f"{module_name} needs {required_count} samples with "
                    f"is_malicious={is_malicious}; found {len(sample_ids)}"
                )
            generator.shuffle(sample_ids)
            sample_pools[module_name][is_malicious] = sample_ids

    # Freeze case membership before inference, including benign controls.
    for pattern in itertools.product((False, True), repeat=3):
        repetitions = 3
        if not any(pattern):
            repetitions = 9
        for _ in range(repetitions):
            case_samples = {}
            for module_name, is_malicious in zip(MODULE_NAMES, pattern):
                case_samples[module_name] = sample_pools[module_name][is_malicious].pop()
            manifest["cases"].append({
                "case_id": f"case-{len(manifest['cases']) + 1:02d}",
                "samples": case_samples,
                "is_malicious": any(pattern),
            })
    return manifest


def request_result(base_url: str, path: str, payload: dict, module_name: str) -> dict:
    start = time.perf_counter()
    try:
        result = post_json(base_url, path, payload)
        if module_name == "claude":
            validated = OrchestrationResponse.model_validate(result)
            if not set(validated.contributing_modules).issubset(payload):
                raise ValueError("Claude cited an unavailable module")
            highest_rank = max(SEVERITY_RANK[item["severity"]] for item in payload.values())
            if SEVERITY_RANK[validated.severity] < highest_rank:
                raise ValueError("Claude returned a severity below the highest detector")
        else:
            OrchestrationRequest.model_validate({module_name: result})
        record = {"result": result, "error": None}
    except (RuntimeError, ValueError) as error:
        record = {"result": None, "error": str(error)}
    record["requested_at"] = datetime.now(timezone.utc).isoformat()
    record["request_elapsed_ms"] = round((time.perf_counter() - start) * 1000, 2)
    record["base_url"] = base_url
    return record


def run_evaluation(manifest: dict, results: dict, run_dir: Path, base_url: str,
                   with_claude: bool, retry_failed: bool) -> None:
    for module_name in MODULE_NAMES:
        samples = manifest["samples"][module_name]
        for position, (sample_id, sample) in enumerate(samples.items(), start=1):
            previous = results["detectors"][module_name].get(sample_id)
            if previous and (previous["result"] is not None or not retry_failed):
                continue
            record = request_result(
                base_url, DETECTOR_PATHS[module_name], sample["request"], module_name
            )
            if previous:
                record["previous_attempts"] = previous.get("previous_attempts", []) + [
                    {key: value for key, value in previous.items() if key != "previous_attempts"}
                ]
            results["detectors"][module_name][sample_id] = record
            save_json(run_dir / "results.json", results)
            status = "failed"
            if record["result"] is not None:
                status = record["result"]["severity"]
            print(f"{module_name} {position}/{len(samples)}: {status}", flush=True)

    if not with_claude:
        return

    for case in manifest["cases"]:
        case_id = case["case_id"]
        previous = results["triage"].get(case_id)
        if previous and (previous["result"] is not None or not retry_failed):
            continue
        detections = get_case_detections(case, results)
        if len(detections) != len(MODULE_NAMES):
            print(f"{case_id}: Claude skipped because a detector is unavailable", flush=True)
            continue
        record = request_result(base_url, ORCHESTRATION_PATH, detections, "claude")
        if previous:
            record["previous_attempts"] = previous.get("previous_attempts", []) + [
                {key: value for key, value in previous.items() if key != "previous_attempts"}
            ]
        results["triage"][case_id] = record
        save_json(run_dir / "results.json", results)
        status = "failed"
        if record["result"] is not None:
            status = record["result"]["severity"]
        print(f"{case_id}: Claude {status}", flush=True)


def get_case_detections(case: dict, results: dict) -> dict:
    detections = {}
    for module_name, sample_id in case["samples"].items():
        record = results["detectors"][module_name].get(sample_id)
        if record and record["result"] is not None:
            detections[module_name] = record["result"]
    return detections


def binary_metrics(truth: list[bool], predictions: list[bool]) -> dict:
    counts = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for actual, predicted in zip(truth, predictions, strict=True):
        if actual:
            key = "tp" if predicted else "fn"
        else:
            key = "fp" if predicted else "tn"
        counts[key] += 1
    ratios = {
        "accuracy": (counts["tp"] + counts["tn"], len(truth)),
        "precision": (counts["tp"], counts["tp"] + counts["fp"]),
        "recall": (counts["tp"], counts["tp"] + counts["fn"]),
        "false_positive_rate": (counts["fp"], counts["fp"] + counts["tn"]),
        "f1": (2 * counts["tp"], 2 * counts["tp"] + counts["fp"] + counts["fn"]),
    }
    metrics = {"evaluated": len(truth), **counts}
    for name, (numerator, denominator) in ratios.items():
        metrics[name] = None
        if denominator:
            metrics[name] = numerator / denominator
    return metrics


def write_summary(manifest: dict, results: dict, run_dir: Path) -> dict:
    summary = {"positive_decision": manifest["positive_decision"], "samples": {}}
    for module_name, samples in manifest["samples"].items():
        truth = []
        predictions = []
        failed = 0
        for sample_id, sample in samples.items():
            record = results["detectors"][module_name].get(sample_id)
            if record is None:
                continue
            if record["result"] is None:
                failed += 1
                continue
            truth.append(sample["is_malicious"])
            predictions.append(record["result"]["severity"] != "LOW")
        summary["samples"][module_name] = {
            "total": len(samples),
            "failed": failed,
            "pending": len(samples) - len(truth) - failed,
            **binary_metrics(truth, predictions),
        }

    rows = []
    for case in manifest["cases"]:
        detections = get_case_detections(case, results)
        row = {"case_id": case["case_id"], "is_malicious": case["is_malicious"]}
        for module_name, sample_id in case["samples"].items():
            sample = manifest["samples"][module_name][sample_id]
            row[f"{module_name}_source_row"] = sample_id
            row[f"{module_name}_source_label"] = sample["source_label"]
            row[f"{module_name}_severity"] = detections.get(module_name, {}).get("severity", "")
        row["max_severity"] = ""
        if len(detections) == len(MODULE_NAMES):
            row["max_severity"] = max(
                (result["severity"] for result in detections.values()), key=SEVERITY_RANK.get
            )
        record = results["triage"].get(case["case_id"])
        row["claude_severity"] = ""
        row["triage_status"] = "not_run"
        if record:
            row["triage_status"] = "failed"
            if record["result"] is not None:
                row["claude_severity"] = record["result"]["severity"]
                row["triage_status"] = "complete"
        rows.append(row)

    complete_rows = [row for row in rows if row["max_severity"]]
    paired_rows = [row for row in complete_rows if row["claude_severity"]]
    summary["cases"] = {
        "total": len(rows),
        "complete_detectors": len(complete_rows),
        "complete_triage": len(paired_rows),
        "failed_triage": sum(row["triage_status"] == "failed" for row in rows),
        "triage_not_run": sum(row["triage_status"] == "not_run" for row in rows),
    }
    # Compare systems on identical completed cases; disclose missing cases separately.
    systems = ("email", "url", "network", "max")
    for cohort_name, cohort, compared_systems in (
        ("detector_baselines", complete_rows, systems),
        ("paired_with_claude", paired_rows, systems + ("claude",)),
    ):
        summary[cohort_name] = {}
        for system in compared_systems:
            summary[cohort_name][system] = binary_metrics(
                [row["is_malicious"] for row in cohort],
                [row[f"{system}_severity"] != "LOW" for row in cohort],
            )
    summary["severity_agreement_with_max"] = {
        "compared": len(paired_rows),
        "equal": sum(row["max_severity"] == row["claude_severity"] for row in paired_rows),
    }
    save_json(run_dir / "summary.json", summary)
    with (run_dir / "cases.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--resume", type=Path, help="Reuse a saved run, including successful responses")
    parser.add_argument("--with-claude", action="store_true", help="Request up to 30 paid triage reports")
    parser.add_argument("--retry-failed", action="store_true", help="Retry saved failed requests")
    args = parser.parse_args()

    try:
        if args.resume:
            if args.input_dir:
                parser.error("--resume uses the saved inputs; omit --input-dir")
            run_dir = args.resume.resolve()
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        else:
            input_dir = args.input_dir
            if input_dir is None:
                input_dir = PROJECT_ROOT / "triage reports" / "eval_inputs"
                if not input_dir.is_dir():
                    input_dir = PROJECT_ROOT / "triage reports" / "evaluation_inputs"
            manifest = create_manifest(input_dir)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_dir = OUTPUT_ROOT / timestamp
            run_dir.mkdir(parents=True)
            results = {"detectors": {name: {} for name in MODULE_NAMES}, "triage": {}}
            save_json(run_dir / "manifest.json", manifest)
            save_json(run_dir / "results.json", results)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Could not prepare evaluation: {error}")

    print(f"Run folder: {run_dir}", flush=True)
    if args.with_claude:
        print("Claude enabled: incomplete cases may use API credit.", flush=True)
    exit_code = 0
    try:
        run_evaluation(manifest, results, run_dir, args.base_url, args.with_claude, args.retry_failed)
    except KeyboardInterrupt:
        print("Stopped. Completed requests are saved; use --resume to continue.")
        exit_code = 130
    finally:
        summary = write_summary(manifest, results, run_dir)

    for module_name, metrics in summary["samples"].items():
        print(
            f"{module_name}: {metrics['evaluated']}/{metrics['total']} completed; "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
        )
        if metrics["failed"] or metrics["pending"]:
            exit_code = exit_code or 1
    if args.with_claude and summary["cases"]["complete_triage"] < len(manifest["cases"]):
        exit_code = exit_code or 1
    print(f"Saved summary.json and cases.csv in {run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
