#!/usr/bin/env python3
"""Plot Section 3 oracle-vs-logged-policy regression bias."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.plot_policy_confounding_bias_ladder import (  # noqa: E402
    DEFAULT_BRANCH_CSV,
    DEFAULT_LADDER,
    bool_series,
    ladder_mask,
)
from scripts.experiments.run_section3_policy_assignment_experiment import (  # noqa: E402
    DEFAULT_OUTPUT_CSV as DEFAULT_POLICY_PANEL_CSV,
    DEFAULT_SUMMARY_JSON as DEFAULT_POLICY_SUMMARY_JSON,
    POLICY_LABELS,
    POLICY_ORDER,
)
from simulation.publication_plot_style import (  # noqa: E402
    apply_top_journal_style,
    concept_color,
    finalize_axes,
    metadata_from_context,
    save_figure,
)


DEFAULT_OUTPUT_STEM = REPO_ROOT / "figures" / "main" / "fig_section3_policy_regression_bias"
DEFAULT_SOURCE_CSV = (
    REPO_ROOT / "data" / "figure_source" / "main" / "fig_section3_policy_regression_bias.csv"
)

POLICY_STYLE = {
    "randomized": {
        "label": "random policy",
        "color": concept_color("cloned"),
        "linestyle": (0, (4, 1.5)),
        "marker": "s",
    },
    "signal_confounded": {
        "label": "confounded policy",
        "color": concept_color("naive"),
        "linestyle": "--",
        "marker": "^",
    },
    "rl_style": {
        "label": "RL-style policy",
        "color": concept_color("terminal"),
        "linestyle": "-.",
        "marker": "D",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-csv", type=Path, default=DEFAULT_BRANCH_CSV)
    parser.add_argument("--policy-panel-csv", type=Path, default=DEFAULT_POLICY_PANEL_CSV)
    parser.add_argument("--policy-summary-json", type=Path, default=DEFAULT_POLICY_SUMMARY_JSON)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--liquidity-scale", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--ladder", default=",".join(str(x) for x in DEFAULT_LADDER))
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260522)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_ladder(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("ladder cannot be empty")
    return tuple(sorted(values))


def fit_ols(q: np.ndarray, y: np.ndarray) -> dict[str, float]:
    design = np.column_stack([np.ones_like(q), q])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ np.array([intercept, slope])
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def load_oracle_branches(args: argparse.Namespace, ladder: tuple[float, ...]) -> pd.DataFrame:
    df = pd.read_csv(args.branch_csv)
    for col in ["liquidity_scale", "horizon", "q_economic_intended", "impact_by_horizon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = df.loc[
        np.isclose(df["liquidity_scale"], args.liquidity_scale)
        & df["horizon"].eq(args.horizon)
        & ladder_mask(df["q_economic_intended"], ladder)
    ].copy()
    supported = bool_series(rows["support_ok"]) & ~bool_series(rows["clipped_flag"])
    if "branch_completed_flag" in rows:
        supported &= bool_series(rows["branch_completed_flag"])
    rows = rows.loc[supported].copy()
    rows["q"] = rows["q_economic_intended"].astype(float).mask(
        np.isclose(rows["q_economic_intended"], 0.0), 0.0
    )
    if rows.empty:
        raise ValueError("No supported oracle rows remain after Section 3 filters.")
    return rows


def load_policy_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not args.policy_panel_csv.exists():
        raise FileNotFoundError(
            f"Missing {args.policy_panel_csv}. Run "
            "scripts/experiments/run_section3_policy_assignment_experiment.py first."
        )
    panel = pd.read_csv(args.policy_panel_csv)
    for col in ["q", "impact_by_horizon", "latent_sign"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    panel = panel.loc[panel["policy_id"].isin(POLICY_ORDER)].dropna(
        subset=["q", "impact_by_horizon", "latent_sign"]
    )
    summary = (
        json.loads(args.policy_summary_json.read_text(encoding="utf-8"))
        if args.policy_summary_json.exists()
        else {}
    )
    return panel, summary


def oracle_curve(rows: pd.DataFrame, ladder: tuple[float, ...]) -> tuple[pd.DataFrame, dict[str, float]]:
    grouped = rows.groupby("q", observed=True)["impact_by_horizon"]
    curve = grouped.agg(terminal_mean="mean", terminal_std="std", n_oracle="size").reindex(ladder)
    curve["terminal_se"] = curve["terminal_std"] / np.sqrt(curve["n_oracle"].astype(float))
    curve = curve.reset_index().rename(columns={"index": "q"})
    curve["q"] = curve["q"].astype(float)
    fit = fit_ols(curve["q"].to_numpy(float), curve["terminal_mean"].to_numpy(float))
    curve["oracle_fit_pred"] = fit["intercept"] + fit["slope"] * curve["q"].to_numpy(float)
    return curve, fit


def policy_fits(panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for policy_id, group in panel.groupby("policy_id", sort=False):
        out[policy_id] = fit_ols(
            group["q"].to_numpy(float),
            group["impact_by_horizon"].to_numpy(float),
        )
    return out


def bootstrap_policy_slopes(panel: pd.DataFrame, reps: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for policy_idx, policy_id in enumerate(POLICY_ORDER):
        rows = panel.loc[panel["policy_id"].eq(policy_id)].copy()
        clusters = rows["clone_id"].dropna().unique()
        by_cluster = {cluster: group for cluster, group in rows.groupby("clone_id", sort=False)}
        vals: list[float] = []
        for _ in range(reps):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            boot = pd.concat([by_cluster[cluster] for cluster in sampled], ignore_index=True)
            vals.append(fit_ols(boot["q"].to_numpy(float), boot["impact_by_horizon"].to_numpy(float))["slope"])
        arr = np.asarray(vals, dtype=float)
        out[policy_id] = {
            "ci_low": float(np.nanquantile(arr, 0.025)),
            "ci_high": float(np.nanquantile(arr, 0.975)),
            "bootstrap_reps": int(reps),
            "bootstrap_seed": int(seed + policy_idx),
        }
    return out


def ratio_rows(
    oracle_fit: dict[str, float],
    fits: dict[str, dict[str, float]],
    boot: dict[str, dict[str, float]],
) -> pd.DataFrame:
    target = float(oracle_fit["slope"])
    rows: list[dict[str, Any]] = [
        {
            "panel": "slope_ratio",
            "estimator": "Oracle benchmark",
            "policy_id": "oracle",
            "estimate": target,
            "target": target,
            "ratio_to_target": 1.0,
            "ratio_ci_low": np.nan,
            "ratio_ci_high": np.nan,
        }
    ]
    for policy_id in POLICY_ORDER:
        estimate = float(fits[policy_id]["slope"])
        rows.append(
            {
                "panel": "slope_ratio",
                "estimator": f"{POLICY_LABELS[policy_id]} OLS",
                "policy_id": policy_id,
                "estimate": estimate,
                "target": target,
                "ratio_to_target": estimate / target,
                "ratio_ci_low": float(boot[policy_id]["ci_low"]) / target,
                "ratio_ci_high": float(boot[policy_id]["ci_high"]) / target,
            }
        )
    return pd.DataFrame(rows)


def assignment_rows(panel: pd.DataFrame) -> pd.DataFrame:
    out = (
        panel.groupby(["policy_id", "policy_label", "latent_direction", "latent_sign"], observed=True)
        .agg(q_mean=("q", "mean"), q_se=("q", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))))
        .reset_index()
    )
    out["panel"] = "assignment_by_latent"
    return out


def build_source(
    curve: pd.DataFrame,
    panel: pd.DataFrame,
    ratios: pd.DataFrame,
    assignments: pd.DataFrame,
    fits: dict[str, dict[str, float]],
    oracle_fit: dict[str, float],
) -> pd.DataFrame:
    curve_src = curve.copy()
    curve_src["panel"] = "oracle_curve"
    for policy_id in POLICY_ORDER:
        fit = fits[policy_id]
        curve_src[f"{policy_id}_ols_pred"] = (
            fit["intercept"] + fit["slope"] * curve_src["q"].to_numpy(float)
        )
        counts = (
            panel.loc[panel["policy_id"].eq(policy_id)]
            .groupby("q", observed=True)
            .size()
            .rename(f"n_{policy_id}")
        )
        curve_src = curve_src.merge(counts, on="q", how="left")
        curve_src[f"n_{policy_id}"] = curve_src[f"n_{policy_id}"].fillna(0).astype(int)

    points = panel.copy()
    points["panel"] = "policy_points"
    points["oracle_slope"] = float(oracle_fit["slope"])
    ratios["oracle_slope"] = float(oracle_fit["slope"])
    assignments["oracle_slope"] = float(oracle_fit["slope"])
    return pd.concat([curve_src, points, ratios, assignments], ignore_index=True, sort=False)


def jitter(values: pd.Series, seed: int, scale: float = 1.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return values.to_numpy(float) + rng.normal(0.0, scale, len(values))


def make_plot(
    curve: pd.DataFrame,
    panel: pd.DataFrame,
    ratios: pd.DataFrame,
    assignments: pd.DataFrame,
    fits: dict[str, dict[str, float]],
) -> Any:
    apply_top_journal_style()
    fig = plt.figure(figsize=(7.15, 5.45), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.92], width_ratios=[1.18, 0.82])
    ax_curve = fig.add_subplot(gs[0, :])
    ax_ratio = fig.add_subplot(gs[1, 0])
    ax_assign = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.12, right=0.985, top=0.92, bottom=0.11, wspace=0.36, hspace=0.62)

    q = curve["q"].to_numpy(float)
    terminal_mean = curve["terminal_mean"].to_numpy(float)
    terminal_se = curve["terminal_se"].fillna(0.0).to_numpy(float)
    oracle_color = concept_color("oracle")
    ax_curve.plot(q, terminal_mean, color=oracle_color, linewidth=1.45, label="oracle response")
    ax_curve.fill_between(
        q,
        terminal_mean - 1.96 * terminal_se,
        terminal_mean + 1.96 * terminal_se,
        color=oracle_color,
        alpha=0.13,
        linewidth=0.0,
    )
    ax_curve.plot(
        q,
        curve["oracle_fit_pred"].to_numpy(float),
        color=oracle_color,
        linewidth=0.95,
        linestyle=(0, (1.6, 1.4)),
        alpha=0.9,
        label="oracle linear fit",
    )
    for policy_id in POLICY_ORDER:
        style = POLICY_STYLE[policy_id]
        fit = fits[policy_id]
        ax_curve.plot(
            q,
            fit["intercept"] + fit["slope"] * q,
            color=style["color"],
            linewidth=1.15,
            linestyle=style["linestyle"],
            label=style["label"],
        )
    ax_curve.axhline(0.0, color="#6b7280", linewidth=0.75, alpha=0.72)
    ax_curve.axvline(0.0, color="#6b7280", linewidth=0.75, alpha=0.72)
    ax_curve.set_title("A. Price-impact response curves")
    ax_curve.set_xlabel(r"signed order size $Q$")
    ax_curve.set_ylabel("terminal response")
    ax_curve.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.6,
        columnspacing=0.9,
        borderpad=0.1,
    )
    finalize_axes(ax_curve)
    ax_curve.grid(True, alpha=0.28)

    plot = ratios.copy()
    y_pos = np.arange(len(plot))[::-1]
    color_map = {
        "oracle": oracle_color,
        "randomized": POLICY_STYLE["randomized"]["color"],
        "signal_confounded": POLICY_STYLE["signal_confounded"]["color"],
        "rl_style": POLICY_STYLE["rl_style"]["color"],
    }
    marker_map = {"oracle": "o", "randomized": "s", "signal_confounded": "^", "rl_style": "D"}
    ax_ratio.axvline(1.0, color=oracle_color, linewidth=1.45, linestyle=(0, (3, 1.6)))
    ax_ratio.text(
        1.04,
        len(plot) - 1.05,
        "oracle causal slope",
        ha="left",
        va="center",
        fontsize=6.8,
        color=oracle_color,
    )
    for ypos, row in zip(y_pos, plot.itertuples()):
        color = color_map[row.policy_id]
        marker = marker_map[row.policy_id]
        ratio = float(row.ratio_to_target)
        if np.isfinite(row.ratio_ci_low) and np.isfinite(row.ratio_ci_high):
            ax_ratio.errorbar(
                ratio,
                ypos,
                xerr=[[ratio - float(row.ratio_ci_low)], [float(row.ratio_ci_high) - ratio]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=0.9,
                capsize=2.0,
                markersize=4.1,
            )
        else:
            ax_ratio.plot(ratio, ypos, marker=marker, color=color, markersize=4.1)
    ax_ratio.set_title("B. Slope vs oracle")
    ax_ratio.set_xlabel("ratio")
    ax_ratio.set_yticks(
        y_pos,
        ["Oracle", "Random", "Confounded", "RL-style"],
    )
    ax_ratio.set_xlim(0.45, 2.25)
    ax_ratio.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    finalize_axes(ax_ratio)
    ax_ratio.grid(True, axis="x")

    x_positions = {policy_id: idx for idx, policy_id in enumerate(POLICY_ORDER)}
    for _, row in assignments.iterrows():
        policy_id = str(row["policy_id"])
        sign = float(row["latent_sign"])
        color = "#1f77b4" if sign > 0 else "#7f1d1d"
        marker = "^" if sign > 0 else "v"
        offset = -0.08 if sign > 0 else 0.08
        ax_assign.errorbar(
            x_positions[policy_id] + offset,
            float(row["q_mean"]),
            yerr=1.96 * float(row["q_se"]),
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=1.8,
            markersize=4.0,
        )
    handles = [
        Line2D([0], [0], marker="^", linestyle="none", color="#1f77b4", markersize=4.0, label="latent buy"),
        Line2D([0], [0], marker="v", linestyle="none", color="#7f1d1d", markersize=4.0, label="latent sell"),
    ]
    ax_assign.legend(handles=handles, loc="upper left", frameon=False, handlelength=0.8, borderpad=0.1)
    ax_assign.axhline(0.0, color="#6b7280", linewidth=0.75, alpha=0.72)
    ax_assign.set_title("C. Policy selection")
    ax_assign.set_ylabel(r"mean selected $Q$")
    ax_assign.set_xticks(
        [x_positions[policy_id] for policy_id in POLICY_ORDER],
        ["Random", "Conf.", "RL"],
        rotation=0,
    )
    ax_assign.set_xlim(-0.55, len(POLICY_ORDER) - 0.45)
    finalize_axes(ax_assign)
    ax_assign.grid(True, axis="y")
    return fig


def main() -> None:
    args = parse_args()
    ladder = parse_ladder(args.ladder)
    oracle_rows = load_oracle_branches(args, ladder)
    panel, summary = load_policy_panel(args)
    curve, oracle_fit = oracle_curve(oracle_rows, ladder)
    fits = policy_fits(panel)
    boot = bootstrap_policy_slopes(panel, args.bootstrap_reps, args.bootstrap_seed)
    ratios = ratio_rows(oracle_fit, fits, boot)
    assignments = assignment_rows(panel)

    source = build_source(curve, panel, ratios.copy(), assignments.copy(), fits, oracle_fit)
    args.source_csv.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.source_csv, index=False)

    fig = make_plot(curve, panel, ratios, assignments, fits)
    metadata = metadata_from_context(
        command="python scripts/experiments/plot_section3_policy_regression_bias.py",
        input_paths=[rel(args.branch_csv), rel(args.policy_panel_csv), rel(args.policy_summary_json)],
        figure_label="fig:section3_policy_regression_bias",
        claim=(
            "A fresh one-branch-per-clone randomized policy stays close to the oracle "
            "price-impact response, while confounded and RL-style logged policies "
            "produce biased response-curve slopes."
        ),
        evidence_type="stored cloned simulator intervention ladder with freshly generated logged policy panels",
        filters=(
            f"liquidity_scale={args.liquidity_scale:g}; horizon={args.horizon}; "
            "support_ok and branch_completed_flag true; clipped_flag false; "
            "all logged policies select exactly one branch per clone"
        ),
        extra={
            "source_data": rel(args.source_csv),
            "source_data_path": rel(args.source_csv),
            "policy_panel_summary": summary,
            "oracle_slope": float(oracle_fit["slope"]),
            "policy_slopes": {policy_id: float(fits[policy_id]["slope"]) for policy_id in POLICY_ORDER},
            "ratio_summary": ratios.to_dict(orient="records"),
            "bootstrap_reps": args.bootstrap_reps,
            "bootstrap_seed": args.bootstrap_seed,
        },
    )
    outputs: list[str] = []
    for suffix in (".pdf", ".png"):
        out = args.output_stem.with_suffix(suffix)
        out_metadata = dict(metadata)
        out_metadata["output_path"] = rel(out)
        save_figure(fig, out, out_metadata)
        outputs.append(rel(out))
    plt.close(fig)
    print(
        json.dumps(
            {
                "outputs": outputs,
                "ratio_summary": ratios[["estimator", "ratio_to_target"]].to_dict(orient="records"),
                "source_data": rel(args.source_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
