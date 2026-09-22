import json

from scripts import run_triage_cases


def test_run_case_sends_only_detector_results_to_orchestration(monkeypatch):
    case = {
        "id": "mixed_signals",
        "description": "Synthetic test case",
        "email": "A synthetic email with a URL",
        "url": "https://example.test/verify",
        "network_file": "models/network_module/network_attack_request.json",
    }
    requests = []

    def fake_post_json(base_url, path, payload):
        requests.append((path, payload))
        if path == run_triage_cases.ORCHESTRATION_PATH:
            return {"severity": "HIGH"}
        return {"severity": "HIGH", "model_version": "test-model"}

    monkeypatch.setattr(run_triage_cases, "post_json", fake_post_json)

    report = run_triage_cases.run_case(case, "http://127.0.0.1:8000", True)

    assert len(requests) == 4
    assert requests[0][1] == {"text": case["email"]}
    assert requests[1][1] == {"url": case["url"]}
    assert len(requests[2][1]["features"]) == 78
    assert requests[3][1] == report["detections"]
    assert case["email"] not in json.dumps(requests[3][1])
    assert case["url"] not in json.dumps(requests[3][1])
    assert report["triage"]["severity"] == "HIGH"
    assert report["triage_requested"] is True


def test_run_case_skips_orchestration_after_detector_failure(monkeypatch):
    case = {
        "id": "benign_control",
        "description": "Synthetic test case",
        "email": "A routine message",
        "url": "https://example.test/about",
        "network_file": "models/network_module/network_benign_request.json",
    }
    requests = []

    def fake_post_json(base_url, path, payload):
        requests.append(path)
        if path == run_triage_cases.DETECTOR_PATHS["email"]:
            raise RuntimeError("Email detector failed")
        return {"severity": "LOW"}

    monkeypatch.setattr(run_triage_cases, "post_json", fake_post_json)

    report = run_triage_cases.run_case(case, "http://127.0.0.1:8000", True)

    assert len(requests) == 3
    assert report["errors"]["email"] == "Email detector failed"
    assert report["triage"] is None


def test_run_case_does_not_call_claude_by_default(monkeypatch):
    case = {
        "id": "benign_control",
        "description": "Synthetic test case",
        "email": "A routine message",
        "url": "https://example.test/about",
        "network_file": "models/network_module/network_benign_request.json",
    }
    requests = []

    def fake_post_json(base_url, path, payload):
        requests.append(path)
        return {"severity": "LOW"}

    monkeypatch.setattr(run_triage_cases, "post_json", fake_post_json)

    report = run_triage_cases.run_case(case, "http://127.0.0.1:8000", False)

    assert len(requests) == 3
    assert report["triage"] is None
    assert report["errors"] == {}
    assert report["triage_requested"] is False
