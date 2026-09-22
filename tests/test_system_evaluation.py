import json
from collections import Counter

import pytest

from scripts import evaluate_system
from scripts import run_triage_cases


@pytest.fixture
def exported_samples(tmp_path):
    for module_name in evaluate_system.MODULE_NAMES:
        samples = []
        for row_index in range(40):
            is_malicious = row_index >= 20
            if module_name == "email":
                label = int(is_malicious)
                request = {"text": f"Synthetic email {row_index}"}
            elif module_name == "url":
                label = "phishing" if is_malicious else "benign"
                request = {"url": f"https://example.test/{row_index}"}
            else:
                label = "DoS Hulk" if is_malicious else "BENIGN"
                request = {"features": [float(row_index)] * 78}
            samples.append({
                "source_row_index": row_index,
                "source_label": label,
                "is_malicious": is_malicious,
                "request": request,
            })
        path = tmp_path / f"{module_name}_evaluation_samples.json"
        path.write_text(json.dumps({"module": module_name, "samples": samples}), encoding="utf-8")
    return tmp_path


def test_manifest_is_reproducible_and_does_not_reuse_source_rows(exported_samples):
    manifest = evaluate_system.create_manifest(exported_samples)
    repeated = evaluate_system.create_manifest(exported_samples)
    assert manifest["cases"] == repeated["cases"]
    patterns = Counter()
    for case in manifest["cases"]:
        pattern = tuple(
            manifest["samples"][name][case["samples"][name]]["is_malicious"]
            for name in evaluate_system.MODULE_NAMES
        )
        patterns[pattern] += 1
        assert case["is_malicious"] == any(pattern)
    assert len(patterns) == 8
    assert patterns[(False, False, False)] == 9
    assert all(count == 3 for pattern, count in patterns.items() if any(pattern))
    for module_name in evaluate_system.MODULE_NAMES:
        selected = [case["samples"][module_name] for case in manifest["cases"]]
        assert len(selected) == len(set(selected)) == 30


def test_manifest_rejects_labels_that_disagree(exported_samples):
    path = exported_samples / "url_evaluation_samples.json"
    export = json.loads(path.read_text(encoding="utf-8"))
    export["samples"][0]["is_malicious"] = True
    path.write_text(json.dumps(export), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        evaluate_system.create_manifest(exported_samples)


def test_metrics_include_false_alarms_and_missed_attacks():
    metrics = evaluate_system.binary_metrics([False, False, True, True], [False, True, False, True])
    assert {key: metrics[key] for key in ("tn", "fp", "fn", "tp")} == {
        "tn": 1, "fp": 1, "fn": 1, "tp": 1,
    }
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert evaluate_system.binary_metrics([], [])["accuracy"] is None


def test_resume_keeps_labels_out_of_requests_and_reuses_successes(exported_samples, monkeypatch):
    manifest = evaluate_system.create_manifest(exported_samples)
    results = {"detectors": {name: {} for name in evaluate_system.MODULE_NAMES}, "triage": {}}
    requests = []

    def fake_request_result(base_url, path, payload, module_name):
        requests.append((module_name, payload))
        return {"result": {"severity": "LOW"}, "error": None}

    monkeypatch.setattr(evaluate_system, "request_result", fake_request_result)
    monkeypatch.setattr(evaluate_system, "save_json", lambda path, content: None)
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", False, False)
    assert len(requests) == 120
    for module_name, payload in requests:
        request_key = {"email": "text", "url": "url", "network": "features"}[module_name]
        assert set(payload) == {request_key}
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", True, False)
    assert len(requests) == 150
    for module_name, payload in requests[120:]:
        assert module_name == "claude"
        assert payload == {name: {"severity": "LOW"} for name in evaluate_system.MODULE_NAMES}
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", True, False)
    assert len(requests) == 150


def test_summary_excludes_incomplete_cases_from_all_paired_baselines(exported_samples):
    manifest = evaluate_system.create_manifest(exported_samples)
    results = {"detectors": {name: {} for name in evaluate_system.MODULE_NAMES}, "triage": {}}
    for module_name, samples in manifest["samples"].items():
        for sample_id in samples:
            results["detectors"][module_name][sample_id] = {
                "result": {"severity": "LOW"}, "error": None,
            }
    first_case = manifest["cases"][0]
    results["detectors"]["network"][first_case["samples"]["network"]] = {
        "result": None, "error": "API unavailable",
    }
    second_case = manifest["cases"][1]
    results["triage"][second_case["case_id"]] = {"result": {"severity": "HIGH"}, "error": None}
    summary = evaluate_system.write_summary(manifest, results, exported_samples)
    assert summary["samples"]["network"]["failed"] == 1
    assert summary["cases"]["complete_detectors"] == 29
    assert summary["cases"]["complete_triage"] == 1
    for metrics in summary["detector_baselines"].values():
        assert metrics["evaluated"] == 29
    for metrics in summary["paired_with_claude"].values():
        assert metrics["evaluated"] == 1
    assert summary["paired_with_claude"]["claude"]["fp"] == 1
    assert summary["paired_with_claude"]["max"]["fp"] == 0


def test_invalid_api_response_is_recorded_as_failure(monkeypatch):
    monkeypatch.setattr(evaluate_system, "post_json", lambda *args: {"severity": "HIGH"})
    record = evaluate_system.request_result("http://localhost", "/analyze", {"text": "test"}, "email")
    assert record["result"] is None
    assert record["error"]


def test_failed_detector_skips_paid_call_until_explicit_retry(exported_samples, monkeypatch):
    manifest = evaluate_system.create_manifest(exported_samples)
    results = {"detectors": {name: {} for name in evaluate_system.MODULE_NAMES}, "triage": {}}
    failed_case = manifest["cases"][0]
    failed_sample_id = failed_case["samples"]["network"]
    failed_request = manifest["samples"]["network"][failed_sample_id]["request"]
    requests = []
    should_fail = True

    def fake_request_result(base_url, path, payload, module_name):
        requests.append(module_name)
        if should_fail and module_name == "network" and payload == failed_request:
            return {"result": None, "error": "Timeout"}
        return {"result": {"severity": "LOW"}, "error": None}

    monkeypatch.setattr(evaluate_system, "request_result", fake_request_result)
    monkeypatch.setattr(evaluate_system, "save_json", lambda path, content: None)
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", True, False)
    assert len(requests) == 149
    assert failed_case["case_id"] not in results["triage"]
    should_fail = False
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", True, False)
    assert len(requests) == 149
    evaluate_system.run_evaluation(manifest, results, exported_samples, "http://localhost", True, True)
    assert requests[149:] == ["network", "claude"]
    assert results["triage"][failed_case["case_id"]]["result"] is not None
    assert results["detectors"]["network"][failed_sample_id]["previous_attempts"][0]["error"] == "Timeout"


def test_request_timeout_becomes_a_recordable_error(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("Request timed out")

    monkeypatch.setattr(run_triage_cases, "urlopen", timeout)
    with pytest.raises(RuntimeError, match="Request timed out"):
        run_triage_cases.post_json("http://localhost", "/analyze", {})
