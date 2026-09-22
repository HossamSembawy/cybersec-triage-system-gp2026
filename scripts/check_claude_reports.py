"""Recheck selected Claude reports using saved detector results."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.evaluate_system import get_case_detections, request_result
from scripts.run_triage_cases import DEFAULT_BASE_URL, ORCHESTRATION_PATH

DEFAULT_CASE_IDS = ("case-17", "case-23", "case-24", "case-30")


def check_reports(run_dir: Path, case_ids: list[str] | None, base_url: str) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in manifest["cases"]}
    if case_ids is None:
        case_ids = list(cases)

    unknown_ids = sorted(set(case_ids) - set(cases))
    if unknown_ids:
        raise ValueError(f"Unknown case IDs: {', '.join(unknown_ids)}")

    checked = {}
    output_dir = run_dir / "prompt_checks"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = output_dir / f"claude_{timestamp}.json"

    for case_id in case_ids:
        detections = get_case_detections(cases[case_id], results)
        if len(detections) != 3:
            raise ValueError(f"Detector result missing for {case_id}")

        response = request_result(base_url, ORCHESTRATION_PATH, detections, "claude")
        checked[case_id] = {
            "source_results": detections,
            "previous_report": results["triage"].get(case_id),
            "new_report": response,
        }
        output_path.write_text(json.dumps(checked, indent=2) + "\n", encoding="utf-8")

        if response["error"]:
            print(f"{case_id}: {response['error']}", flush=True)
            raise RuntimeError(f"Claude check stopped after {case_id} failed; saved {output_path}")
        else:
            report = response["result"]
            print(f"{case_id}: {report['severity']}", flush=True)
            print(report["evidence_summary"], flush=True)
            print(flush=True)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-id", action="append", dest="case_ids")
    selection.add_argument("--all-cases", action="store_true")
    args = parser.parse_args()

    case_ids = None if args.all_cases else args.case_ids or list(DEFAULT_CASE_IDS)
    output_path = check_reports(args.run_dir, case_ids, args.base_url)
    print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
