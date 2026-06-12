from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "plots" / "figure_manifest.yaml"


def upsert_manifest(entries: list[dict[str, Any]]) -> None:
    """Update the lightweight public figure manifest.

    The private repository has a broader figure builder.  The public branch only
    needs manifest bookkeeping for the figures reproduced by
    scripts/reproduce_paper_figures.py.
    """
    current: dict[str, Any] = {"schema_version": 1, "figures": []}
    if MANIFEST.exists():
        current = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_id = {item["figure_id"]: item for item in current.get("figures", [])}
    for entry in entries:
        by_id[entry["figure_id"]] = entry
    payload = {"schema_version": 1, "figures": list(by_id.values())}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
