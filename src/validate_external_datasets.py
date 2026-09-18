"""Generate an inspectable data-quality report for the two external benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multitopology_adapter import load_sample, read_manifest, validate_stratified_split


def build_report(multi_root: Path, ic_root: Path) -> dict:
    manifest_path = multi_root / "metadata" / "manifest.csv"
    split_report = validate_stratified_split(
        read_manifest(manifest_path), multi_root / "metadata" / "summary.json"
    )
    sample_reports = []
    for path in sorted((multi_root / "TestData").glob("*/**/data.h5")):
        sample = load_sample(path)
        fields = sample["raw_fields"]
        sample_reports.append(
            {
                "path": path.relative_to(multi_root).as_posix(),
                "topology": str(sample["attributes"]["version"]),
                "instances": int(len(sample["row_ids"])),
                "field_keys": sorted(fields),
                "feature_shape": list(sample["features"].shape),
                "target_shape": list(sample["target"].shape),
                "temperature_range_c": [
                    float(fields["temperature"].min()),
                    float(fields["temperature"].max()),
                ],
            }
        )
    expected_topologies = set(split_report["topology_split"])
    sampled_topologies = {row["topology"] for row in sample_reports}
    source_status = {
        "code_present": (ic_root / "run.py").is_file(),
        "official_metrics_present": (ic_root / "utils" / "metrics.py").is_file(),
        "dataset_contract_present": (ic_root / "docs" / "DATASETS.md").is_file(),
        "s5_manifest_present": (ic_root / "level5" / "manifest.json").is_file(),
        "checkpoint_downloaded": False,
    }
    return {
        "multi_topology": {
            "manifest": split_report,
            "sampled_all_topologies": sampled_topologies == expected_topologies,
            "sample_count": len(sample_reports),
            "samples": sample_reports,
        },
        "ic_thermbench": source_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-root", type=Path, required=True)
    parser.add_argument("--ic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.multi_root, args.ic_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

