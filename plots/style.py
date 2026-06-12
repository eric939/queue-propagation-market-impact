from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
    "black": "#111111",
    "light_gray": "#E6E6E6",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.4,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.2,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def add_target_line(ax, value: float, label: str = "target") -> None:
    ax.axhline(value, color=COLORS["black"], linewidth=1.0, label=label, zorder=1)


def support_rate_annotation(ax, support_rate: float | None = None, clipping_rate: float | None = None) -> None:
    parts = []
    if support_rate is not None:
        parts.append(f"support {support_rate:.1%}")
    if clipping_rate is not None:
        parts.append(f"clip {clipping_rate:.1%}")
    if parts:
        ax.text(
            0.02,
            0.98,
            ", ".join(parts),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": COLORS["light_gray"], "pad": 2},
        )


def coef_plot(ax, data, *, x_col: str = "estimate", label_col: str = "estimator", target_col: str = "target") -> None:
    labels = list(data[label_col])
    y = np.arange(len(data))[::-1]
    validity = [str(v).lower() for v in data.get("validity", ["valid"] * len(data))]
    colors = [COLORS["blue"] if v == "valid" or v.startswith("valid_") else COLORS["red"] for v in validity]
    markers = ["o" if v == "valid" or v.startswith("valid_") else "s" for v in validity]
    for i, (_, row) in enumerate(data.iterrows()):
        yi = y[i]
        lo = row.get("ci_low", row[x_col])
        hi = row.get("ci_high", row[x_col])
        ax.errorbar(
            row[x_col],
            yi,
            xerr=[[row[x_col] - lo], [hi - row[x_col]]],
            fmt=markers[i],
            color=colors[i],
            ecolor=colors[i],
            capsize=2,
            markersize=4,
            linewidth=1.0,
        )
    if target_col in data:
        targets = data[target_col].dropna().unique()
        if len(targets) == 1:
            ax.axvline(float(targets[0]), color=COLORS["black"], linewidth=1.0, label="target")
        else:
            ax.axvline(0.0, color=COLORS["gray"], linestyle="--", linewidth=0.8, label="reference")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=0.7)


def target_recovery_plot(ax, data, *, label_col: str = "panel_name") -> None:
    y = np.arange(len(data))[::-1]
    ax.errorbar(
        data["estimate"],
        y,
        xerr=[data["estimate"] - data["ci_low"], data["ci_high"] - data["estimate"]],
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        capsize=2,
        markersize=4,
        label="estimate",
    )
    ax.scatter(data["target"], y, marker="|", s=90, color=COLORS["black"], label="branch target", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(data[label_col])
    ax.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=0.7)


def predicted_vs_observed_plot(ax, data) -> None:
    families = list(data["proxy_family"].unique()) if "proxy_family" in data else ["proxy"]
    markers = ["o", "s", "^", "D", "v"]
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"], COLORS["red"]]
    for i, family in enumerate(families):
        sub = data[data["proxy_family"].eq(family)] if "proxy_family" in data else data
        ax.scatter(
            sub["predicted_target_shift"],
            sub["observed_target_shift"],
            marker=markers[i % len(markers)],
            color=palette[i % len(palette)],
            label=str(family).replace("_", " "),
            s=24,
            alpha=0.85,
        )
    lo = float(min(data["predicted_target_shift"].min(), data["observed_target_shift"].min()))
    hi = float(max(data["predicted_target_shift"].max(), data["observed_target_shift"].max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS["black"], linewidth=0.9, label="45-degree")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)


def write_figure_metadata(path: Path, metadata: dict[str, Any], *, write_legacy: bool = True) -> None:
    return None


def save_figure(fig, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    metadata = dict(metadata)
    metadata.pop("write_png", None)
    metadata.pop("write_legacy_metadata", None)
    metadata["output_pdf"] = rel(path)
    metadata["output_hashes"] = {"pdf": _hash(path)}
    write_figure_metadata(path, metadata)
    plt.close(fig)
