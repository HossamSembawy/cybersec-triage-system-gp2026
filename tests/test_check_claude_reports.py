import json

import pytest

from scripts import check_claude_reports


def test_check_reports_reuses_detectors_and_preserves_original_run(tmp_path, monkeypatch):
    manifest = {
        "cases": [{
            "case_id": "case-17",
            "samples": {"email": "1", "url": "2", "network": "3"},
        }]
    }
    detections = {
        "email": {"1": {"result": {"severity": "LOW"}}},
        "url": {"2": {"result": {"severity": "HIGH"}}},
        "network": {"3": {"result": {"severity": "MEDIUM"}}},
    }
    results = {
        "detectors": detections,
        "triage": {"case-17": {"result": {"severity": "HIGH", "evidence_summary": "Old"}}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    requests = []

    def fake_request(base_url, path, payload, module_name):
        requests.append((base_url, path, payload, module_name))
        return {"result": {"severity": "HIGH", "evidence_summary": "New"}, "error": None}

    monkeypatch.setattr(check_claude_reports, "request_result", fake_request)

    output_path = check_claude_reports.check_reports(
        tmp_path, ["case-17"], "http://127.0.0.1:8000"
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(requests) == 1
    assert requests[0][2] == {
        "email": {"severity": "LOW"},
        "url": {"severity": "HIGH"},
        "network": {"severity": "MEDIUM"},
    }
    assert saved["case-17"]["previous_report"]["result"]["evidence_summary"] == "Old"
    assert saved["case-17"]["new_report"]["result"]["evidence_summary"] == "New"
    assert json.loads(results_path.read_text(encoding="utf-8")) == results


def test_check_reports_rejects_unknown_case_before_call(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text('{"cases": []}', encoding="utf-8")
    (tmp_path / "results.json").write_text('{"detectors": {}, "triage": {}}', encoding="utf-8")
    monkeypatch.setattr(
        check_claude_reports,
        "request_result",
        lambda *arguments: pytest.fail("An unknown case must not call Claude"),
    )

    with pytest.raises(ValueError, match="Unknown case IDs"):
        check_claude_reports.check_reports(tmp_path, ["case-99"], "http://127.0.0.1:8000")


def test_check_reports_selects_every_case_when_no_ids_are_supplied(tmp_path, monkeypatch):
    manifest = {
        "cases": [
            {"case_id": case_id, "samples": {"email": "1", "url": "2", "network": "3"}}
            for case_id in ("case-01", "case-02")
        ]
    }
    results = {
        "detectors": {
            module: {sample_id: {"result": {"severity": "LOW"}}}
            for module, sample_id in (("email", "1"), ("url", "2"), ("network", "3"))
        },
        "triage": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    calls = []

    def fake_request(base_url, path, payload, module_name):
        calls.append(payload)
        return {"result": {"severity": "LOW", "evidence_summary": "No elevated signal."}, "error": None}

    monkeypatch.setattr(check_claude_reports, "request_result", fake_request)

    output_path = check_claude_reports.check_reports(tmp_path, None, "http://127.0.0.1:8000")

    assert len(calls) == 2
    assert set(json.loads(output_path.read_text(encoding="utf-8"))) == {"case-01", "case-02"}


def test_check_reports_stops_after_failed_call(tmp_path, monkeypatch):
    manifest = {
        "cases": [
            {"case_id": case_id, "samples": {"email": "1", "url": "2", "network": "3"}}
            for case_id in ("case-01", "case-02")
        ]
    }
    results = {
        "detectors": {
            module: {sample_id: {"result": {"severity": "LOW"}}}
            for module, sample_id in (("email", "1"), ("url", "2"), ("network", "3"))
        },
        "triage": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(results), encoding="utf-8")
    calls = []

    def fake_request(base_url, path, payload, module_name):
        calls.append(payload)
        return {"result": None, "error": "HTTP 503"}

    monkeypatch.setattr(check_claude_reports, "request_result", fake_request)

    with pytest.raises(RuntimeError, match="stopped after case-01 failed"):
        check_claude_reports.check_reports(tmp_path, None, "http://127.0.0.1:8000")

    assert len(calls) == 1
    saved_files = list((tmp_path / "prompt_checks").glob("*.json"))
    assert len(saved_files) == 1
    assert list(json.loads(saved_files[0].read_text(encoding="utf-8"))) == ["case-01"]
