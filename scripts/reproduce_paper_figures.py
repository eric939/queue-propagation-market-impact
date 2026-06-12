#!/usr/bin/env python3
"""Rebuild the active paper figures from the public reproduction bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures" / "main"
MANIFEST_OUT = ROOT / "artifacts" / "reproduction" / "figure_rebuild_manifest.json"

PAPER_FIGURES = (
    "fig_section3_policy_regression_bias",
    "fig_oracle_decomposition_atlas",
    "fig_parent_schedule_bundle",
    "fig_functional_civ_curve_validation",
    "fig_civ_validity_frontier",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-policy-panel",
        action="store_true",
        help="Reuse the existing Section 3 logged-policy panel instead of regenerating it.",
    )
    parser.add_argument(
        "--skip-existing-functional",
        action="store_true",
        help="Leave the functional-CIV bundle untouched if the PDF already exists.",
    )
    return parser.parse_args()


def run_script(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def build_evidence_figures() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from plots import build_evidence_synthesis_figures as evidence
    from plots.build_main_figures import upsert_manifest

    evidence.setup_style()
    entries = [
        evidence.figure_oracle_decomposition_atlas(),
        evidence.figure_parent_schedule_bundle(),
        evidence.figure_civ_validity_frontier(),
    ]
    upsert_manifest(entries)


def build_manifest() -> dict[str, Any]:
    figures: list[dict[str, Any]] = []
    missing: list[str] = []
    for figure_id in PAPER_FIGURES:
        pdf = FIGURE_DIR / f"{figure_id}.pdf"
        png = FIGURE_DIR / f"{figure_id}.png"
        source = ROOT / "data" / "figure_source" / "main" / f"{figure_id}.csv"
        metadata = pdf.with_suffix(".pdf.metadata.json")
        if not pdf.exists():
            missing.append(rel(pdf))
            continue
        row: dict[str, Any] = {
            "figure_id": figure_id,
            "pdf": rel(pdf),
            "pdf_sha256": sha256(pdf),
        }
        if png.exists():
            row["png"] = rel(png)
            row["png_sha256"] = sha256(png)
        if source.exists():
            row["source_csv"] = rel(source)
            row["source_csv_sha256"] = sha256(source)
        if metadata.exists():
            row["metadata_json"] = rel(metadata)
        figures.append(row)
    if missing:
        raise FileNotFoundError("Missing rebuilt figure outputs: " + ", ".join(missing))

    payload = {
        "schema_version": 1,
        "figures": figures,
        "entrypoint": "python3 scripts/reproduce_paper_figures.py",
    }
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    if not args.skip_policy_panel:
        run_script("scripts/experiments/run_section3_policy_assignment_experiment.py")
    run_script("scripts/experiments/plot_section3_policy_regression_bias.py")
    build_evidence_figures()
    functional_args = ["scripts/experiments/build_functional_civ_curve_bundle.py"]
    if not args.skip_existing_functional:
        functional_args.append("--force")
    run_script(*functional_args)
    payload = build_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
