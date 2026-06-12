#!/usr/bin/env python3
"""Bundle cloned-grid and observed-panel functional-CIV curve checks.

The bundled figure replaces separate main-text CIV curve figures with a
matched 2-by-4 layout.  It reads existing figure-source/audit CSVs and writes
the manuscript figure without rerunning the simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.publication_plot_style import (  # noqa: E402
    add_panel_label,
    apply_top_journal_style,
    concept_color,
)


CLONED_CURVES = REPO_ROOT / "data" / "figure_source" / "main" / "fig_functional_civ_curve_validation.csv"
OBSERVED_CURVES = REPO_ROOT / "artifacts" / "simulated_observed_valid_instrument" / "curves.csv"
OBSERVED_SUMMARY = REPO_ROOT / "artifacts" / "simulated_observed_valid_instrument" / "summary.csv"
OUTPUT_PDF = REPO_ROOT / "figures" / "main" / "fig_functional_civ_curve_validation.pdf"
OUTPUT_PNG = OUTPUT_PDF.with_suffix(".png")

LIQUIDITY_SETTINGS: tuple[tuple[float, str], ...] = (
    (1.5, "Low liquidity"),
    (3.5, "Medium liquidity"),
    (10.0, "High liquidity"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloned-curves", type=Path, default=CLONED_CURVES)
    parser.add_argument("--observed-curves", type=Path, default=OBSERVED_CURVES)
    parser.add_argument("--observed-summary", type=Path, default=OBSERVED_SUMMARY)
    parser.add_argument("--output-pdf", type=Path, default=OUTPUT_PDF)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cloned(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "row_type",
        "liquidity_scale",
        "q",
        "interventional_benchmark",
        "sieve_civ_curve",
        "scalar_civ_line",
        "same_basis_no_iv_line",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    grid = df.loc[df["row_type"].eq("grid")].copy()
    grid["design"] = "cloned_grid"
    grid["oracle"] = pd.to_numeric(grid["interventional_benchmark"], errors="coerce")
    grid["functional_civ"] = pd.to_numeric(grid["sieve_civ_curve"], errors="coerce")
    grid["scalar_projection"] = pd.to_numeric(grid["scalar_civ_line"], errors="coerce")
    grid["same_basis_no_iv"] = pd.to_numeric(grid["same_basis_no_iv_line"], errors="coerce")
    return grid[["design", "liquidity_scale", "q", "oracle", "functional_civ", "scalar_projection", "same_basis_no_iv"]]


def load_cloned_loss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    summary = df.loc[df["row_type"].eq("panel_summary")].copy()
    if summary.empty:
        raise ValueError(f"{path} has no panel_summary rows for loss panels")
    return pd.DataFrame(
        {
            "liquidity_scale": pd.to_numeric(summary["liquidity_scale"], errors="coerce"),
            "functional_civ": pd.to_numeric(summary["crossfit_cell_rmse"], errors="coerce"),
            "scalar_projection": pd.to_numeric(summary["scalar_rmse_intervention_cells"], errors="coerce"),
            "same_basis_no_iv": pd.to_numeric(summary["same_basis_no_iv_support_rmse"], errors="coerce"),
        }
    )


def load_observed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {
        "row_type",
        "liquidity_scale",
        "q",
        "oracle_curve",
        "functional_iv",
        "scalar_iv",
        "confounded_no_iv",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    grid = df.loc[df["row_type"].eq("grid")].copy()
    grid["design"] = "observed_panel"
    grid["oracle"] = pd.to_numeric(grid["oracle_curve"], errors="coerce")
    grid["functional_civ"] = pd.to_numeric(grid["functional_iv"], errors="coerce")
    grid["scalar_projection"] = pd.to_numeric(grid["scalar_iv"], errors="coerce")
    grid["same_basis_no_iv"] = pd.to_numeric(grid["confounded_no_iv"], errors="coerce")
    return grid[["design", "liquidity_scale", "q", "oracle", "functional_civ", "scalar_projection", "same_basis_no_iv"]]


def load_observed_loss(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    summary = pd.read_csv(path)
    return pd.DataFrame(
        {
            "liquidity_scale": pd.to_numeric(summary["liquidity_scale"], errors="coerce"),
            "functional_civ": pd.to_numeric(summary["functional_iv_rmse_cells"], errors="coerce"),
            "scalar_projection": pd.to_numeric(summary["scalar_iv_rmse_cells"], errors="coerce"),
            "same_basis_no_iv": pd.to_numeric(summary["confounded_no_iv_rmse_cells"], errors="coerce"),
        }
    )


def plot_panel(ax: plt.Axes, rows: pd.DataFrame, *, title: str, panel_label: str, show_ylabel: bool, show_xlabel: bool) -> None:
    rows = rows.sort_values("q")
    q = rows["q"].to_numpy(dtype=float)
    oracle = rows["oracle"].to_numpy(dtype=float)
    civ = rows["functional_civ"].to_numpy(dtype=float)
    scalar = rows["scalar_projection"].to_numpy(dtype=float)
    no_iv = rows["same_basis_no_iv"].to_numpy(dtype=float)

    ax.axhline(0.0, color="#9ca3af", linewidth=0.7)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.7)
    ax.plot(q, oracle, color=concept_color("oracle"), linestyle="-", linewidth=2.35, label="cloned oracle", zorder=3)
    ax.plot(q, civ, color=concept_color("civ"), linestyle="-", linewidth=1.65, label="functional CIV", zorder=4)
    ax.plot(
        q,
        scalar,
        color="#6b7280",
        linestyle=":",
        linewidth=1.35,
        label="scalar projection",
        zorder=3,
    )
    finite_no_iv = np.isfinite(no_iv)
    ax.plot(
        q[finite_no_iv],
        no_iv[finite_no_iv],
        color=concept_color("naive"),
        linestyle="--",
        linewidth=1.55,
        alpha=0.90,
        label="same-basis no-IV",
        zorder=2,
    )
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel("signed dose Q")
    if show_ylabel:
        ax.set_ylabel("residual after mechanics (ticks)")
    ax.grid(True, axis="both")
    ax.tick_params(axis="both", labelsize=7.3)
    add_panel_label(ax, panel_label)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    y_candidates = np.concatenate([oracle, civ, scalar, no_iv[finite_no_iv]])
    y_candidates = y_candidates[np.isfinite(y_candidates)]
    if y_candidates.size:
        y_min = float(np.min(y_candidates))
        y_max = float(np.max(y_candidates))
        pad = max(0.08, 0.10 * (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)


def plot_loss_panel(ax: plt.Axes, loss: pd.DataFrame, *, title: str, panel_label: str) -> None:
    loss = loss.copy().sort_values("liquidity_scale")
    y_pos = np.arange(len(LIQUIDITY_SETTINGS), dtype=float)
    label_lookup = {liquidity: label.replace(" liquidity", "") for liquidity, label in LIQUIDITY_SETTINGS}
    method_specs = [
        ("functional_civ", "CIV", concept_color("civ"), -0.22),
        ("scalar_projection", "scalar", "#6b7280", 0.0),
        ("same_basis_no_iv", "no-IV", concept_color("naive"), 0.22),
    ]
    loss_lookup = {
        float(row.liquidity_scale): row
        for row in loss.itertuples(index=False)
    }
    for key, label, color, offset in method_specs:
        values = np.asarray(
            [float(getattr(loss_lookup[liquidity], key)) for liquidity, _ in LIQUIDITY_SETTINGS],
            dtype=float,
        )
        ax.barh(y_pos + offset, values, height=0.18, color=color, alpha=0.88, label=label)
        for y_value, value in zip(y_pos + offset, values):
            text = f"{value:.3f}" if value >= 0.01 else f"{value:.3g}"
            ax.text(value * 1.10, y_value, text, va="center", ha="left", fontsize=6.3)
    ax.set_xscale("log")
    max_value = max(
        float(loss[["functional_civ", "scalar_projection", "same_basis_no_iv"]].to_numpy().max()),
        0.01,
    )
    ax.set_xlim(0.003, max(3.0, max_value * 2.4))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([label_lookup[liquidity] for liquidity, _ in LIQUIDITY_SETTINGS])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("RMSE")
    ax.grid(True, axis="x")
    ax.tick_params(axis="both", labelsize=6.9)
    add_panel_label(ax, panel_label)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    output_pdf = args.output_pdf
    output_png = output_pdf.with_suffix(".png")
    if output_pdf.exists() and not args.force:
        raise FileExistsError(f"{output_pdf} exists; pass --force to overwrite.")

    cloned = load_cloned(args.cloned_curves)
    cloned_loss = load_cloned_loss(args.cloned_curves)
    observed = load_observed(args.observed_curves)
    observed_loss = load_observed_loss(args.observed_summary)
    summary = pd.read_csv(args.observed_summary) if args.observed_summary.exists() else pd.DataFrame()

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    apply_top_journal_style()
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.65, 5.15),
        constrained_layout=False,
        sharex="col",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.90]},
    )
    fig.subplots_adjust(left=0.075, right=0.990, bottom=0.115, top=0.850, wspace=0.40, hspace=0.55)

    for col_idx, (liquidity, label) in enumerate(LIQUIDITY_SETTINGS):
        cloned_rows = cloned.loc[np.isclose(cloned["liquidity_scale"], liquidity)]
        observed_rows = observed.loc[np.isclose(observed["liquidity_scale"], liquidity)]
        plot_panel(
            axes[0, col_idx],
            cloned_rows,
            title=label,
            panel_label=chr(ord("A") + col_idx),
            show_ylabel=col_idx == 0,
            show_xlabel=False,
        )
        plot_panel(
            axes[1, col_idx],
            observed_rows,
            title="",
            panel_label=chr(ord("E") + col_idx),
            show_ylabel=col_idx == 0,
            show_xlabel=True,
        )

    plot_loss_panel(axes[0, 3], cloned_loss, title="Loss", panel_label="D")
    plot_loss_panel(axes[1, 3], observed_loss, title="", panel_label="H")

    axes[0, 0].set_ylabel("Cloned grid\nresidual after mechanics (ticks)")
    axes[1, 0].set_ylabel("Observed panel\nresidual after mechanics (ticks)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = ["cloned oracle", "functional CIV", "scalar projection", "same-basis no-IV"]
    lookup = {label: handle for handle, label in zip(handles, labels)}
    fig.legend(
        [lookup[label] for label in order],
        order,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.985),
        frameon=False,
        ncol=4,
        columnspacing=1.4,
        handlelength=2.5,
        handletextpad=0.55,
        borderpad=0.1,
        fontsize=7.5,
    )
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure_id": "fig_functional_civ_curve_validation",
        "figure_label": "fig:functional_civ_curve_validation",
        "claim": (
            "Functional residual-CIV recovers the cloned residual propagation curve "
            "in the theorem-optimized cloned grid and in an observed-panel simulator "
            "positive control with one randomized realized arm per pseudo-state."
        ),
        "evidence_type": "bundled_cloned_and_observed_panel_functional_civ",
        "command": "python3 scripts/experiments/build_functional_civ_curve_bundle.py --force",
        "input_paths": [
            rel(args.cloned_curves),
            rel(args.observed_curves),
            rel(args.observed_summary),
        ],
        "output_pdf": rel(output_pdf),
        "output_png": rel(output_png),
        "series": ["cloned oracle", "functional CIV", "scalar projection", "same-basis no-IV"],
        "loss_panels": {
            "top": "RMSE against cloned-grid oracle for cloned-grid estimator",
            "bottom": "RMSE against cloned-grid oracle for observed-panel randomized estimator",
        },
        "design_rows": {
            "top": "full cloned-grid estimator with cloned-state structure",
            "bottom": "observed-panel randomized estimator with one realized arm per pseudo-state",
        },
        "observed_panel_summary": summary.to_dict(orient="records") if not summary.empty else [],
        "output_hashes": {
            "pdf": file_hash(output_pdf),
            "png": file_hash(output_png),
        },
    }
    output_pdf.with_suffix(".pdf.metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = build_bundle(args)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
