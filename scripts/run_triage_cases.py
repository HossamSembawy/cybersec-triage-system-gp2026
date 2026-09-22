"""Run saved triage cases against the local API."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "scripts" / "triage_cases.json"
REPORTS_DIR = PROJECT_ROOT / "triage reports"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120

DETECTOR_PATHS = {
    "email": "/api/v1/email/analyze",
    "url": "/api/v1/url/analyze",
    "network": "/api/v1/network/analyze",
}
ORCHESTRATION_PATH = "/api/v1/orchestration/analyze"


def post_json(base_url: str, path: str, payload: dict) -> dict:
    request = Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            try:
                return json.load(response)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path} returned invalid JSON") from error
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach {base_url}: {error.reason}") from error
    except (TimeoutError, ConnectionError) as error:
        raise RuntimeError(f"Request to {path} failed: {error}") from error


def run_case(case: dict, base_url: str, with_claude: bool) -> dict:
    network_path = PROJECT_ROOT / case["network_file"]
    with network_path.open(encoding="utf-8") as network_file:
        network_request = json.load(network_file)

    detector_requests = {
        "email": {"text": case["email"]},
        "url": {"url": case["url"]},
        "network": network_request,
    }
    report = {
        "case_id": case["id"],
        "description": case["description"],
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "detections": {},
        "errors": {},
        "triage_requested": with_claude,
        "triage": None,
        "triage_error": None,
    }

    for module_name, path in DETECTOR_PATHS.items():
        try:
            report["detections"][module_name] = post_json(
                base_url, path, detector_requests[module_name]
            )
        except RuntimeError as error:
            report["errors"][module_name] = str(error)

    # Avoid a paid, partial triage call when any detector has failed.
    if with_claude and not report["errors"]:
        try:
            report["triage"] = post_json(
                base_url, ORCHESTRATION_PATH, report["detections"]
            )
        except RuntimeError as error:
            report["triage_error"] = str(error)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--case", help="Run one case by ID; default runs all cases")
    parser.add_argument(
        "--with-claude",
        action="store_true",
        help="Request a combined triage report; this uses Anthropic API credit",
    )
    args = parser.parse_args()

    with CASES_FILE.open(encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    if args.case:
        cases = [case for case in cases if case["id"] == args.case]
        if not cases:
            parser.error(f"Unknown case: {args.case}")

    REPORTS_DIR.mkdir(exist_ok=True)
    failed_cases = 0

    for case in cases:
        try:
            report = run_case(case, args.base_url, args.with_claude)
        except (OSError, ValueError) as error:
            print(f"{case['id']}: could not load the network flow: {error}")
            failed_cases += 1
            continue

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_path = REPORTS_DIR / f"{case['id']}-{timestamp}.json"
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")

        detector_count = len(report["detections"])
        triage_status = "skipped"
        if report["triage"]:
            triage_status = report["triage"]["severity"]
        elif report["triage_error"]:
            triage_status = "failed"

        print(f"{case['id']}: {detector_count}/3 detectors, triage {triage_status}")
        for module_name, result in report["detections"].items():
            print(f"  {module_name}: {result['prediction']} ({result['severity']})")
        print(f"Saved {report_path}")

        if report["errors"] or report["triage_error"]:
            failed_cases += 1

    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
