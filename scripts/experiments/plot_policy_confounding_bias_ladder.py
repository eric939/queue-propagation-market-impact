#!/usr/bin/env python3
"""Plot random-policy and confounded-policy regressions against an oracle benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.publication_plot_style import (  # noqa: E402
    apply_top_journal_style,
    concept_color,
    finalize_axes,
    figure_size,
    metadata_from_context,
    save_figure,
)


DEFAULT_BRANCH_CSV = (
    REPO_ROOT
    / "data"
    / "market_impact_study"
    / "strategic_groundtruth"
    / "extended_intervention_ladder"
    / "latest"
    / "data"
    / "one_shot_branch_rows.csv"
)
DEFAULT_OUTPUT_STEM = REPO_ROOT / "figures" / "main" / "fig_policy_confounding_bias_l1p5"
DEFAULT_SOURCE_CSV = (
    REPO_ROOT / "data" / "figure_source" / "main" / "fig_policy_confounding_bias_l1p5.csv"
)
DEFAULT_LADDER = (-100, -60, -30, -10, 0, 10, 30, 60, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-csv", type=Path, default=DEFAULT_BRANCH_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--liquidity-scale", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument(
        "--ladder",
        default=",".join(str(x) for x in DEFAULT_LADDER),
        help="Comma-separated signed economic order sizes to plot.",
    )
    parser.add_argument(
        "--policy-alignment",
        default="aligned",
        help="Branch alignment used as the confounded policy-observed sample.",
    )
    parser.add_argument(
        "--random-policy-design",
        choices=["complete", "iid"],
        default="complete",
        help=(
            "Randomized-policy replay design. 'complete' assigns a near-balanced "
            "randomized ladder across clone states; 'iid' samples q independently "
            "for each clone."
        ),
    )
    parser.add_argument("--random-policy-seed", type=int, default=20260521)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260521)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def ladder_mask(values: pd.Series, ladder: tuple[float, ...], tol: float = 1e-9) -> pd.Series:
    arr = values.to_numpy(float)
    mask = np.zeros(len(arr), dtype=bool)
    for point in ladder:
        mask |= np.isclose(arr, point, atol=tol, rtol=0.0)
    return pd.Series(mask, index=values.index)


def parse_ladder(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError("ladder cannot be empty")
    return tuple(sorted(values))


def load_filtered_rows(args: argparse.Namespace, ladder: tuple[float, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.branch_csv)
    numeric_cols = [
        "liquidity_scale",
        "horizon",
        "q_economic_intended",
        "impact_by_horizon",
        "f_walk",
        "propagation_residual_by_horizon",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    raw = df.loc[
        np.isclose(df["liquidity_scale"], args.liquidity_scale)
        & (df["horizon"] == args.horizon)
        & ladder_mask(df["q_economic_intended"], ladder)
    ].copy()
    if raw.empty:
        raise ValueError(
            "No rows found for "
            f"liquidity_scale={args.liquidity_scale}, horizon={args.horizon}, ladder={ladder}"
        )

    supported = bool_series(raw["support_ok"]) & ~bool_series(raw["clipped_flag"])
    if "branch_completed_flag" in raw:
        supported &= bool_series(raw["branch_completed_flag"])
    filtered = raw.loc[supported].copy()
    filtered["q"] = filtered["q_economic_intended"].astype(float).mask(
        np.isclose(filtered["q_economic_intended"], 0.0), 0.0
    )
    raw["q"] = raw["q_economic_intended"].astype(float).mask(
        np.isclose(raw["q_economic_intended"], 0.0), 0.0
    )
    if filtered.empty:
        raise ValueError("All selected rows failed support/completion filters.")
    return raw, filtered


def fit_ols(q: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.column_stack([np.ones_like(q), q])
    intercept, slope = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ np.array([intercept, slope])
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }


def fit_intervention_benchmark(rows: pd.DataFrame, ladder: tuple[float, ...]) -> dict[str, float]:
    benchmark_means = (
        rows.groupby("q", observed=True)["impact_by_horizon"].mean().reindex(ladder).dropna()
    )
    return fit_ols(
        benchmark_means.index.to_numpy(float),
        benchmark_means.to_numpy(float),
    )


def draw_random_policy(
    rows: pd.DataFrame,
    ladder: tuple[float, ...],
    *,
    design: str,
    seed: int,
) -> pd.DataFrame:
    """Replay a finite randomized policy from the recorded intervention branches."""
    rng = np.random.default_rng(seed)
    clone_ids = list(rows["clone_id"].drop_duplicates())
    if not clone_ids:
        raise ValueError("No clone states available for random-policy replay.")

    ladder_arr = np.asarray(ladder, dtype=float)
    if design == "complete":
        repeats, remainder = divmod(len(clone_ids), len(ladder_arr))
        assignments = np.tile(ladder_arr, repeats)
        if remainder:
            assignments = np.concatenate(
                [assignments, rng.choice(ladder_arr, size=remainder, replace=False)]
            )
        rng.shuffle(assignments)
    elif design == "iid":
        assignments = rng.choice(ladder_arr, size=len(clone_ids), replace=True)
    else:
        raise ValueError(f"unknown random-policy design: {design}")

    selected: list[pd.Series] = []
    grouped = {clone_id: group for clone_id, group in rows.groupby("clone_id", sort=False)}
    for clone_id, q_value in zip(clone_ids, assignments):
        group = grouped[clone_id]
        match = group[np.isclose(group["q"].to_numpy(float), float(q_value))]
        if match.empty:
            raise ValueError(f"Missing randomized q={q_value:g} for clone {clone_id}")
        selected.append(match.iloc[0])

    out = pd.DataFrame(selected).reset_index(drop=True)
    out["random_policy_seed"] = seed
    out["random_policy_design"] = design
    return out


def cluster_bootstrap_slope(
    rows: pd.DataFrame,
    *,
    ladder: tuple[float, ...],
    mode: str,
    reps: int,
    seed: int,
) -> dict[str, float]:
    clusters = [key for key in rows["clone_id"].dropna().unique()]
    if reps <= 0 or len(clusters) < 2:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_reps": 0}

    rng = np.random.default_rng(seed)
    by_cluster = {cluster: group for cluster, group in rows.groupby("clone_id", sort=False)}
    slopes: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by_cluster[cluster] for cluster in sampled], ignore_index=True)
        if mode == "intervention":
            fit = fit_intervention_benchmark(boot, ladder)
        elif mode == "ols":
            fit = fit_ols(boot["q"].to_numpy(float), boot["impact_by_horizon"].to_numpy(float))
        else:
            raise ValueError(f"unknown bootstrap mode: {mode}")
        slopes.append(float(fit["slope"]))

    lo, hi = np.quantile(np.asarray(slopes, dtype=float), [0.025, 0.975])
    return {"ci_low": float(lo), "ci_high": float(hi), "bootstrap_reps": int(reps)}


def slope_summary(
    *,
    filtered: pd.DataFrame,
    random_policy: pd.DataFrame,
    confounded_policy: pd.DataFrame,
    ladder: tuple[float, ...],
    random_fit: dict[str, float],
    confounded_fit: dict[str, float],
    intervention_fit: dict[str, float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = [
        {
            "estimator": "Oracle benchmark",
            "kind": "oracle",
            "estimate": intervention_fit["slope"],
            **cluster_bootstrap_slope(
                filtered,
                ladder=ladder,
                mode="intervention",
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed,
            ),
        },
        {
            "estimator": "Random-policy OLS",
            "kind": "random",
            "estimate": random_fit["slope"],
            **cluster_bootstrap_slope(
                random_policy,
                ladder=ladder,
                mode="ols",
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed + 1,
            ),
        },
        {
            "estimator": "Alpha-selected OLS",
            "kind": "confounded",
            "estimate": confounded_fit["slope"],
            **cluster_bootstrap_slope(
                confounded_policy,
                ladder=ladder,
                mode="ols",
                reps=args.bootstrap_reps,
                seed=args.bootstrap_seed + 2,
            ),
        },
    ]
    out = pd.DataFrame(rows)
    target = float(intervention_fit["slope"])
    out["ratio_to_oracle"] = out["estimate"] / target
    out["bias_vs_oracle"] = out["estimate"] - target
    return out


def summarize_rows(
    raw: pd.DataFrame,
    filtered: pd.DataFrame,
    random_policy: pd.DataFrame,
    confounded_policy: pd.DataFrame,
    ladder: tuple[float, ...],
    random_fit: dict[str, float],
    confounded_fit: dict[str, float],
) -> pd.DataFrame:
    grouped = filtered.groupby("q", observed=True)
    out = grouped.agg(
        n=("impact_by_horizon", "size"),
        terminal_mean=("impact_by_horizon", "mean"),
        terminal_std=("impact_by_horizon", "std"),
    )
    raw_grouped = raw.groupby("q", observed=True)
    raw_stats = raw_grouped.agg(
        raw_n=("impact_by_horizon", "size"),
        support_rate=("support_ok", lambda s: float(bool_series(s).mean())),
        clipping_rate=("clipped_flag", lambda s: float(bool_series(s).mean())),
    )
    random_counts = random_policy.groupby("q", observed=True).size().rename("n_random_policy")
    confounded_counts = confounded_policy.groupby("q", observed=True).size().rename("n_confounded_policy")

    q_index = pd.Index([float(q) for q in ladder], name="q")
    out = (
        out.reindex(q_index)
        .join(raw_stats.reindex(q_index))
        .join(random_counts.reindex(q_index))
        .join(confounded_counts.reindex(q_index))
    )
    out["n_random_policy"] = out["n_random_policy"].fillna(0).astype(int)
    out["n_confounded_policy"] = out["n_confounded_policy"].fillna(0).astype(int)
    out["terminal_se"] = out["terminal_std"] / np.sqrt(out["n"].astype(float))
    out["oracle_fit_pred"] = np.nan
    out["random_policy_ols_pred"] = (
        random_fit["intercept"] + random_fit["slope"] * out.index.to_numpy(float)
    )
    out["confounded_policy_ols_pred"] = (
        confounded_fit["intercept"] + confounded_fit["slope"] * out.index.to_numpy(float)
    )
    out = out.reset_index()
    return out


def selected_branch_source(
    random_policy: pd.DataFrame,
    confounded_policy: pd.DataFrame,
) -> pd.DataFrame:
    """Store the point-level rows drawn in panel A."""
    keep = [
        "clone_id",
        "q",
        "impact_by_horizon",
        "alignment",
        "latent_direction",
        "liquidity_scale",
        "horizon",
        "random_policy_seed",
        "random_policy_design",
    ]
    frames: list[pd.DataFrame] = []
    for panel, rows in [
        ("random_policy_points", random_policy),
        ("confounded_policy_points", confounded_policy),
    ]:
        cols = [col for col in keep if col in rows.columns]
        out = rows.loc[:, cols].copy()
        out["panel"] = panel
        out["kind"] = "random" if panel.startswith("random") else "confounded"
        frames.append(out)
    return pd.concat(frames, ignore_index=True, sort=False)


def jitter(values: pd.Series, *, seed: int, width: float = 1.45) -> np.ndarray:
    """Deterministic x-jitter for overplotted branch outcomes."""
    rng = np.random.default_rng(seed)
    return values.to_numpy(float) + rng.uniform(-width, width, size=len(values))


def make_plot(
    source: pd.DataFrame,
    slopes: pd.DataFrame,
    random_policy: pd.DataFrame,
    confounded_policy: pd.DataFrame,
    random_fit: dict[str, float],
    confounded_fit: dict[str, float],
    intervention_fit: dict[str, float],
    args: argparse.Namespace,
) -> Any:
    apply_top_journal_style()
    fig, (ax, ax_slope) = plt.subplots(
        1,
        2,
        figsize=figure_size("wide"),
        gridspec_kw={"width_ratios": [2.25, 1.0]},
        constrained_layout=True,
    )
    q = source["q"].to_numpy(float)
    ci = 1.96 * source["terminal_se"].fillna(0.0).to_numpy(float)
    oracle_pred = source["oracle_fit_pred"].to_numpy(float)
    random_pred = source["random_policy_ols_pred"].to_numpy(float)
    confounded_pred = source["confounded_policy_ols_pred"].to_numpy(float)
    ax.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.75)
    ax.axvline(0.0, color="#6b7280", linewidth=0.8, alpha=0.75)
    ax.fill_between(
        q,
        oracle_pred,
        confounded_pred,
        color=concept_color("naive"),
        alpha=0.10,
        linewidth=0.0,
        label="selection-bias gap",
        zorder=0,
    )
    ax.scatter(
        jitter(random_policy["q"], seed=args.random_policy_seed + 17),
        random_policy["impact_by_horizon"],
        s=11,
        marker="o",
        color=concept_color("cloned"),
        alpha=0.20,
        linewidths=0,
        label="_nolegend_",
        zorder=1,
    )
    ax.scatter(
        jitter(confounded_policy["q"], seed=args.random_policy_seed + 31),
        confounded_policy["impact_by_horizon"],
        s=10,
        marker="^",
        color=concept_color("naive"),
        alpha=0.13,
        linewidths=0,
        label="_nolegend_",
        zorder=1,
    )
    ax.plot(
        q,
        oracle_pred,
        color=concept_color("oracle"),
        linewidth=2.0,
        label="_nolegend_",
        zorder=3,
    )
    ax.errorbar(
        q,
        source["terminal_mean"],
        yerr=ci,
        fmt="o",
        color=concept_color("oracle"),
        ecolor="#9ca3af",
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.85,
        markersize=4.1,
        label="cloned mean +/- 95% CI",
        zorder=4,
    )
    ax.plot(
        q,
        random_pred,
        color=concept_color("cloned"),
        linewidth=2.0,
        linestyle=(0, (4, 1.6)),
        marker="s",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="random-policy OLS",
        zorder=5,
    )
    ax.plot(
        q,
        confounded_pred,
        color=concept_color("naive"),
        linewidth=1.9,
        linestyle="--",
        label="alpha-selected OLS",
        zorder=3,
    )

    gap_x = 72.0
    oracle_gap_y = np.interp(gap_x, q, oracle_pred)
    confounded_gap_y = np.interp(gap_x, q, confounded_pred)
    ax.annotate(
        "selection gap",
        xy=(gap_x, 0.5 * (oracle_gap_y + confounded_gap_y)),
        xytext=(34, 3.25),
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": concept_color("naive")},
        color=concept_color("naive"),
        fontsize=7.3,
        ha="left",
        va="center",
    )
    ax.text(
        0.02,
        0.98,
        "faint points: selected one-branch outcomes",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.9,
        color="#4b5563",
    )
    random_label_x = 62.0
    random_label_y = np.interp(random_label_x, q, random_pred)
    ax.annotate(
        "random OLS\n(near oracle)",
        xy=(random_label_x, random_label_y),
        xytext=(38, 0.95),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 0.7,
            "color": concept_color("cloned"),
        },
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.78,
        },
        color=concept_color("cloned"),
        fontsize=7.0,
        ha="left",
        va="center",
    )

    ax.set_title("A. Policy-selected response slices")
    ax.set_xlabel(r"intervention size $q$ (signed lots)")
    ax.set_ylabel("terminal impact (ticks)")
    ax.set_xticks([-100, -60, -30, 0, 30, 60, 100])
    y_values = np.concatenate(
        [
            source["terminal_mean"].to_numpy(float),
            oracle_pred,
            random_pred,
            confounded_pred,
            random_policy["impact_by_horizon"].to_numpy(float),
            confounded_policy["impact_by_horizon"].to_numpy(float),
        ]
    )
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    pad = max(0.35, 0.08 * (y_max - y_min))
    ax.set_xlim(-106, 106)
    ax.set_ylim(y_min - pad, y_max + pad)
    handles, labels = ax.get_legend_handles_labels()
    legend = dict(zip(labels, handles))
    legend_order = [
        "cloned mean +/- 95% CI",
        "random-policy OLS",
        "alpha-selected OLS",
        "selection-bias gap",
    ]
    ax.legend(
        [legend[label] for label in legend_order if label in legend],
        [label for label in legend_order if label in legend],
        loc="lower right",
        ncol=2,
        columnspacing=0.9,
        handlelength=2.3,
    )
    finalize_axes(ax)
    ax.grid(True, axis="both")

    plot_order = ["Oracle benchmark", "Random-policy OLS", "Alpha-selected OLS"]
    slope_plot = slopes.set_index("estimator").loc[plot_order].reset_index()
    y = np.arange(len(slope_plot))[::-1]
    colors = [
        concept_color("oracle"),
        concept_color("cloned"),
        concept_color("naive"),
    ]
    markers = ["o", "o", "^"]
    target = float(intervention_fit["slope"])
    ax_slope.axvline(
        target,
        color=concept_color("oracle"),
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        label="oracle slope",
    )
    for ypos, (_, row), color, marker in zip(y, slope_plot.iterrows(), colors, markers):
        estimate = float(row["estimate"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        if np.isfinite(lo) and np.isfinite(hi):
            ax_slope.errorbar(
                estimate,
                ypos,
                xerr=[[estimate - lo], [hi - estimate]],
                fmt=marker,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.2,
                markersize=4.7,
            )
        else:
            ax_slope.plot(estimate, ypos, marker=marker, color=color, markersize=4.7)
        x_pad = 0.00072
        if row["estimator"] == "Random-policy OLS":
            x_pad = 0.00115
        elif row["estimator"] == "Alpha-selected OLS":
            x_pad = 0.00102
        ax_slope.text(
            estimate + x_pad,
            ypos,
            f"{estimate:.4f} ({float(row['ratio_to_oracle']):.2f}x)",
            ha="left",
            va="center",
            fontsize=7.1,
            color=color,
        )
    ax_slope.set_title("B. Slope with clone bootstrap")
    ax_slope.set_xlabel("ticks per lot")
    ax_slope.set_yticks(y, ["Oracle benchmark", "Randomized replay", "Alpha-selected policy"])
    finite_lows = slope_plot["ci_low"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_highs = slope_plot["ci_high"].replace([np.inf, -np.inf], np.nan).dropna()
    x_min = min(float(finite_lows.min()), float(slope_plot["estimate"].min())) - 0.003
    x_max = max(float(finite_highs.max()), float(slope_plot["estimate"].max())) + 0.004
    ax_slope.set_xlim(max(0.0, x_min), x_max)
    ax_slope.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    finalize_axes(ax_slope)
    ax_slope.grid(True, axis="x")
    return fig


def main() -> None:
    args = parse_args()
    ladder = parse_ladder(args.ladder)
    raw, filtered = load_filtered_rows(args, ladder)
    policy = filtered.loc[
        (filtered["alignment"] == args.policy_alignment) & ~np.isclose(filtered["q"], 0.0)
    ].copy()
    if len(policy) < 2:
        raise ValueError(
            f"Need at least two nonzero {args.policy_alignment!r} policy rows; got {len(policy)}"
        )
    random_policy = draw_random_policy(
        filtered,
        ladder,
        design=args.random_policy_design,
        seed=args.random_policy_seed,
    )

    confounded_fit = fit_ols(
        policy["q"].to_numpy(float),
        policy["impact_by_horizon"].to_numpy(float),
    )
    random_fit = fit_ols(
        random_policy["q"].to_numpy(float),
        random_policy["impact_by_horizon"].to_numpy(float),
    )
    intervention_fit = fit_intervention_benchmark(filtered, ladder)
    slopes = slope_summary(
        filtered=filtered,
        random_policy=random_policy,
        confounded_policy=policy,
        ladder=ladder,
        random_fit=random_fit,
        confounded_fit=confounded_fit,
        intervention_fit=intervention_fit,
        args=args,
    )
    source = summarize_rows(raw, filtered, random_policy, policy, ladder, random_fit, confounded_fit)
    source["oracle_fit_pred"] = (
        intervention_fit["intercept"] + intervention_fit["slope"] * source["q"].to_numpy(float)
    )
    source["random_minus_oracle_pred"] = (
        source["random_policy_ols_pred"] - source["oracle_fit_pred"]
    )
    source["panel"] = "response_curve"
    source = pd.concat(
        [
            source,
            slopes.assign(panel="slope_summary"),
            selected_branch_source(random_policy, policy),
        ],
        ignore_index=True,
        sort=False,
    )

    args.source_csv.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.source_csv, index=False)

    curve_source = source[source["panel"].eq("response_curve")].copy()
    fig = make_plot(
        curve_source,
        slopes,
        random_policy,
        policy,
        random_fit,
        confounded_fit,
        intervention_fit,
        args,
    )
    metadata = metadata_from_context(
        command="python scripts/experiments/plot_policy_confounding_bias_ladder.py",
        input_paths=[rel(args.branch_csv)],
        figure_label="fig:policy_confounding_bias_l1p5",
        claim=(
            "A finite randomized-policy replay stays near the oracle intervention slope, "
            "while alpha-selected policy observations overstate the response."
        ),
        evidence_type="simulator cloned-state intervention with randomized and policy-selected observational slices",
        treatment_ladder=",".join(f"{q:g}" for q in ladder),
        horizon=str(args.horizon),
        filters=(
            f"liquidity_scale={args.liquidity_scale:g}; horizon={args.horizon}; "
            "support_ok and branch_completed_flag true; clipped_flag false; "
            f"random policy is a one-branch-per-clone {args.random_policy_design} "
            f"randomized replay with seed {args.random_policy_seed}; "
            f"confounded policy slice alignment={args.policy_alignment!r}"
        ),
        extra={
            "liquidity_scale": args.liquidity_scale,
            "terminal_horizon": args.horizon,
            "source_data_path": rel(args.source_csv),
            "selected_branch_rows": int(len(filtered)),
            "raw_branch_rows": int(len(raw)),
            "random_policy_rows": int(len(random_policy)),
            "confounded_policy_rows": int(len(policy)),
            "random_policy_ols": random_fit,
            "confounded_policy_ols": confounded_fit,
            "intervention_benchmark_fit": intervention_fit,
            "slope_bootstrap_reps": args.bootstrap_reps,
            "slope_bootstrap_seed": args.bootstrap_seed,
            "random_policy_seed": args.random_policy_seed,
            "random_policy_design": args.random_policy_design,
            "random_policy_assignment_counts": {
                f"{q:g}": int(n)
                for q, n in random_policy.groupby("q", observed=True).size().sort_index().items()
            },
            "slope_summary": slopes.to_dict(orient="records"),
            "random_to_intervention_slope_ratio": float(
                random_fit["slope"] / intervention_fit["slope"]
            ),
            "confounded_to_intervention_slope_ratio": float(
                confounded_fit["slope"] / intervention_fit["slope"]
            ),
            "confounded_policy_selection_rule": (
                f"alignment == {args.policy_alignment!r} and q_economic_intended != 0"
            ),
            "random_policy_selection_rule": (
                f"one q per clone; {args.random_policy_design} randomization over plotted ladder; "
                f"seed={args.random_policy_seed}"
            ),
        },
    )
    outputs: list[str] = []
    for suffix in (".pdf",):
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
                "source_data": rel(args.source_csv),
                "selected_branch_rows": int(len(filtered)),
                "random_policy_rows": int(len(random_policy)),
                "confounded_policy_rows": int(len(policy)),
                "random_policy_ols_slope": random_fit["slope"],
                "confounded_policy_ols_slope": confounded_fit["slope"],
                "intervention_slope": intervention_fit["slope"],
                "random_to_intervention_slope_ratio": random_fit["slope"] / intervention_fit["slope"],
                "confounded_to_intervention_slope_ratio": (
                    confounded_fit["slope"] / intervention_fit["slope"]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
