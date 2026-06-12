#!/usr/bin/env python3
"""Validate that the public branch contains only the code reproduction bundle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_DIRS = {
    ".pytest_cache",
    "__pycache__",
    "docs",
    "latex",
    "notebooks",
    "quality_revision",
    "reports",
    "submission_package",
    "tables",
}
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".bbl",
    ".bcf",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".tex",
    ".toc",
}
FORBIDDEN_NAMES = {
    "main.pdf",
    "main.tex",
    "paper_artifacts_manifest.json",
    "SUBMISSION_MANIFEST.md",
}
REQUIRED_PATHS = {
    "README.md",
    "requirements.txt",
    "scripts/reproduce_paper_figures.py",
    "scripts/check_public_bundle.py",
    "simulation/market_gym.py",
    "config/config.py",
    "limit_order_book/limit_order_book.py",
    "initial_shape/noise_65.npz",
    "initial_shape/noise_flow_65.npz",
    "simulator_upgrades/dock.py",
    "figures/main/fig_section3_policy_regression_bias.pdf",
    "figures/main/fig_oracle_decomposition_atlas.pdf",
    "figures/main/fig_parent_schedule_bundle.pdf",
    "figures/main/fig_functional_civ_curve_validation.pdf",
    "figures/main/fig_civ_validity_frontier.pdf",
}


def tracked_files() -> list[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
        files = [line for line in out.splitlines() if line]
        if files:
            return files
    except Exception:
        pass
    return [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    ]


def has_forbidden_suffix(path: str) -> bool:
    return any(path.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def main() -> int:
    files = tracked_files()
    violations: list[str] = []
    for path in files:
        parts = set(Path(path).parts)
        name = Path(path).name
        if parts.intersection(FORBIDDEN_DIRS):
            violations.append(path)
        elif path.startswith("figures/main/") and not path.endswith(".pdf"):
            violations.append(path)
        elif name in FORBIDDEN_NAMES or has_forbidden_suffix(path):
            violations.append(path)

    missing = sorted(path for path in REQUIRED_PATHS if not (ROOT / path).exists())
    if violations or missing:
        if violations:
            print("Forbidden public-branch files:", file=sys.stderr)
            for path in sorted(violations):
                print(f"  {path}", file=sys.stderr)
        if missing:
            print("Missing required public-branch files:", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
        return 1
    print(f"Public bundle check passed ({len(files)} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
