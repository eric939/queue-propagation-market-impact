"""Top-journal numerical plotting utilities for the QP-SCM paper.

The module is intentionally Matplotlib-only at baseline. Optional style
packages are used when installed, but every public function has a
dependency-free fallback so reproduction scripts remain portable.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.environ.get("MPLBACKEND", "") == "":
    matplotlib.use("Agg")


PALETTE = {
    "oracle": "#111827",
    "target": "#111827",
    "cloned": "#2563eb",
    "walk": "#0f766e",
    "mechanics": "#0f766e",
    "terminal": "#b45309",
    "propagation": "#7c3aed",
    "residual": "#7c3aed",
    "civ": "#0f766e",
    "generated_civ": "#0f766e",
    "naive": "#b91c1c",
    "observational": "#b91c1c",
    "natural_iv_rejected": "#be123c",
    "invalid": "#be123c",
    "native": "#6b7280",
    "stress": "#6b7280",
    "hidden": "#b91c1c",
    "support": "#2563eb",
}

MARKERS = {
    "oracle": "o",
    "target": None,
    "cloned": "o",
    "walk": "s",
    "mechanics": "s",
    "terminal": "o",
    "propagation": "D",
    "residual": "D",
    "civ": "o",
    "generated_civ": "o",
    "naive": "^",
    "observational": "^",
    "natural_iv_rejected": "x",
    "invalid": "x",
    "native": "s",
    "stress": "s",
    "hidden": "o",
    "support": "o",
}

LINESTYLES = {
    "oracle": "-",
    "target": (0, (4, 2)),
    "cloned": "-",
    "walk": "-",
    "mechanics": "-",
    "terminal": "-",
    "propagation": "-",
    "residual": "-",
    "civ": "-",
    "generated_civ": "-",
    "naive": "--",
    "observational": "--",
    "natural_iv_rejected": "--",
    "invalid": "--",
    "native": "-",
    "stress": "-",
    "hidden": "-",
    "support": "-",
}

DISPLAY_NAMES = {
    "oracle": "oracle / cloned target",
    "target": "target",
    "cloned": "cloned intervention",
    "walk": "walk-the-book",
    "mechanics": "mechanics",
    "terminal": "terminal",
    "propagation": "propagation",
    "residual": "residual",
    "civ": "CIV",
    "generated_civ": "generated CIV",
    "naive": "naive observational",
    "observational": "observational",
    "natural_iv_rejected": "rejected natural-IV",
    "invalid": "invalid / rejected",
    "native": "native stress test",
    "stress": "stress diagnostic",
    "hidden": "hidden support",
    "support": "support",
}

FIGURE_SIZES = {
    "single": (3.35, 2.25),
    "single_tall": (3.35, 2.85),
    "double": (6.95, 3.8),
    "double_tall": (6.95, 5.1),
    "appendix": (7.25, 5.6),
    "wide": (7.4, 3.4),
    "six_panel": (7.35, 4.15),
}


def _optional_style_context() -> dict[str, Any]:
    """Return optional rcParams from SciencePlots/tueplots if available."""
    rc: dict[str, Any] = {}
    try:
        import scienceplots  # noqa: F401

        # Do not call plt.style.use here; keep this utility import-light.
    except Exception:
        pass
    try:
        from tueplots import axes, fontsizes

        rc.update(axes.lines())
        rc.update(fontsizes.neurips2023())
    except Exception:
        pass
    return rc


def apply_top_journal_style() -> dict[str, Any]:
    """Apply and return the repository's publication Matplotlib rcParams."""
    rc = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 8.6,
        "axes.titlesize": 9.2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7.6,
        "ytick.labelsize": 7.6,
        "legend.fontsize": 7.4,
        "legend.frameon": False,
        "lines.linewidth": 1.45,
        "lines.markersize": 4.3,
        "patch.linewidth": 0.5,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.75,
    }
    rc.update(_optional_style_context())
    matplotlib.rcParams.update(rc)
    return rc


def figure_size(kind: str) -> tuple[float, float]:
    if kind not in FIGURE_SIZES:
        raise ValueError(f"unknown figure size kind: {kind}")
    return FIGURE_SIZES[kind]


def method_display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


def concept_color(name: str) -> str:
    return PALETTE.get(name, "#2563eb")


def concept_marker(name: str) -> str | None:
    return MARKERS.get(name, "o")


def concept_linestyle(name: str) -> str | tuple[int, tuple[int, int]]:
    return LINESTYLES.get(name, "-")


def format_log_axis(ax: Any) -> None:
    ax.tick_params(which="both", direction="out", length=3.0, width=0.7)
    ax.grid(True, which="major", axis="both")


def add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        fontweight="bold",
    )


def finalize_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7)
    ax.grid(True, axis="y")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def metadata_from_context(
    *,
    command: str,
    input_paths: list[str] | None = None,
    output_path: str | None = None,
    figure_label: str | None = None,
    claim: str | None = None,
    evidence_type: str | None = None,
    seed: int | str | None = None,
    n_episodes: int | str | None = None,
    treatment_ladder: str | None = None,
    horizon: str | None = None,
    intervention_step: str | int | None = None,
    bootstrap: str | None = None,
    filters: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "created_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "git_commit": _git_commit(),
        "command": command,
        "input_paths": input_paths or [],
        "output_path": output_path,
        "figure_label": figure_label,
        "claim": claim,
        "evidence_type": evidence_type,
        "seed": seed,
        "n_episodes": n_episodes,
        "treatment_ladder": treatment_ladder,
        "horizon": horizon,
        "intervention_step": intervention_step,
        "bootstrap": bootstrap,
        "filters": filters,
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_figure_metadata(path: str | Path, metadata: dict[str, Any] | None) -> Path:
    fig_path = Path(path)
    return fig_path


def save_figure(fig: Any, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{uuid.uuid4().hex}{out.suffix}")
    fig.savefig(tmp, bbox_inches="tight")
    tmp.replace(out)
    write_figure_metadata(out, metadata)


def paper_rc() -> dict[str, Any]:
    """Backward-compatible rcParams helper used by older scripts."""
    return apply_top_journal_style()


def estimator_color(name: str) -> str:
    aliases = {
        "OLS": "naive",
        "OLS + controls": "observational",
        "Naive IV": "naive",
        "Linear CIV": "civ",
        "Cross-fit flexible CIV": "civ",
        "CIV, full lags": "civ",
    }
    return concept_color(aliases.get(name, name))


def estimator_marker(name: str) -> str | None:
    aliases = {
        "OLS": "naive",
        "OLS + controls": "observational",
        "Naive IV": "naive",
        "Linear CIV": "civ",
        "Cross-fit flexible CIV": "civ",
        "CIV, full lags": "civ",
    }
    return concept_marker(aliases.get(name, name))


REFERENCE_LINE = {"color": concept_color("target"), "linewidth": 1.1, "linestyle": concept_linestyle("target")}


def add_target_line(ax: Any, value: float, *, label: str = "target") -> Any:
    return ax.axhline(value, label=label, **REFERENCE_LINE)


def label_sim_native_shock() -> str:
    return r"simulator-native shock $x_{\rm sim}$ (+sell)"


def label_economic_abs_parent() -> str:
    return r"absolute parent size $|q|$ (economic)"


def save_pdf_png_svg(fig: Any, stem: Path) -> None:
    """Save the paper-facing figure format.

    The helper name is kept for legacy call sites, but repository outputs are
    now PDF-only to avoid stale parallel figure formats.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = stem.with_suffix(".pdf")
    tmp = out.with_name(f".{out.stem}.{uuid.uuid4().hex}{out.suffix}")
    fig.savefig(tmp, dpi=300, bbox_inches="tight")
    tmp.replace(out)
    write_figure_metadata(out, {"output_path": str(out)})
