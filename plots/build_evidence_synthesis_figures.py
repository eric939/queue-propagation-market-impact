from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from plots.style import COLORS, rel, save_figure, setup_style


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG_DIR = ROOT / "figures" / "main"
SOURCE_DIR = ROOT / "data" / "figure_source" / "main"
TABLE_DIR = ROOT / "artifacts" / "generated_tables"
SCRIPT = "plots/build_evidence_synthesis_figures.py"

PASS = "✓"
WARN = "!"
MISS = "×"

MANUSCRIPT_LABELS = {
    "fig_experiment_design_map": "fig:experiment_design_map",
    "fig_oracle_decomposition_atlas": "fig:section3_oracle_response_components",
    "fig_proxy_error_covariance_identity": "fig:proxy_error_covariance_identity",
    "fig_parent_schedule_bundle": "fig:section3_metaorder_slice_decomposition",
    "fig_schedule_path_decomposition": "fig:section5_metaorder_decomposition",
    "fig_civ_validity_frontier": "fig:civ_validity_frontier",
    "fig_native_schedule_target_heatmap": "fig:native_schedule_target_heatmap",
    "fig_external_lob_boundary": "fig:external_lob_boundary",
    "fig_qpscm_policy_evaluation_stack": "fig:qpscm_policy_evaluation_stack",
}

MANUSCRIPT_ALIASES = {
    "fig_parent_schedule_bundle": ["fig:section5_metaorder_decomposition"],
    "fig_civ_validity_frontier": ["fig:civ_estimator_strength"],
    "fig_native_schedule_target_heatmap": ["fig:native_event_engine_stress_tests"],
}


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    full = ROOT / path
    return pd.read_csv(full) if full.exists() else pd.DataFrame()


def write_source(fig_id: str, df: pd.DataFrame) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / f"{fig_id}.csv"
    if "artifact_provenance" not in df.columns and "source_artifact" not in df.columns and "source_artifacts" not in df.columns:
        df = df.copy()
        df["artifact_provenance"] = SCRIPT
    df.to_csv(path, index=False)
    return path


def _entry(
    *,
    fig_id: str,
    title: str,
    claim: str,
    design_ids: list[str],
    source_artifacts: list[str],
    source_data: Path,
    output: Path,
    evidence_type: str,
    caption: str,
) -> dict[str, Any]:
    entry = {
        "figure_id": fig_id,
        "title": title,
        "main_or_appendix": "main",
        "primary_claim": claim,
        "experiment_design_ids": design_ids,
        "source_artifacts": source_artifacts,
        "source_data": rel(source_data),
        "plot_script": SCRIPT,
        "output_pdf": rel(output),
        "manuscript_label": MANUSCRIPT_LABELS[fig_id],
        "caption_text": caption,
        "evidence_type": evidence_type,
        "allowed_wording": claim,
        "status": "active",
    }
    if fig_id in MANUSCRIPT_ALIASES:
        entry["manuscript_aliases"] = MANUSCRIPT_ALIASES[fig_id]
    return entry


def _save(
    fig,
    *,
    fig_id: str,
    title: str,
    claim: str,
    design_ids: list[str],
    source_artifacts: list[str],
    source_data: Path,
    evidence_type: str,
    caption: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = FIG_DIR / f"{fig_id}.pdf"
    metadata: dict[str, Any] = {
        "figure_id": fig_id,
        "title": title,
        "primary_claim": claim,
        "manuscript_claim": claim,
        "evidence_type": evidence_type,
        "script_path": SCRIPT,
        "source_data": rel(source_data),
        "source_artifact_paths": source_artifacts,
        "experiment_design_ids": design_ids,
        "inference_method": evidence_type,
        "sample_size": int(pd.read_csv(source_data).shape[0]) if source_data.exists() else 0,
        "clusters_or_cells": "see source data",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    save_figure(fig, out, metadata)
    return _entry(
        fig_id=fig_id,
        title=title,
        claim=claim,
        design_ids=design_ids,
        source_artifacts=source_artifacts,
        source_data=source_data,
        output=out,
        evidence_type=evidence_type,
        caption=caption,
    )


def _wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_badge_row(ax, badges: list[tuple[str, str, str]]) -> None:
    ax.set_axis_off()
    n = len(badges)
    for i, (label, value, color) in enumerate(badges):
        x0 = i / n + 0.015
        w = 1 / n - 0.03
        rect = Rectangle((x0, 0.18), w, 0.64, transform=ax.transAxes, facecolor=color, edgecolor="white", lw=1.2)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, 0.58, value, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.text(x0 + w / 2, 0.34, label, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=8.4)


def mean_ci(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    g = df.groupby(group_cols, observed=True)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    se = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - 1.96 * se
    out["ci_high"] = out["mean"] + 1.96 * se
    return out


def cov_ratio(df: pd.DataFrame, z_col: str, y_col: str, x_col: str) -> float:
    z = df[z_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    zc = z - z.mean()
    denom = np.sum(zc * (x - x.mean()))
    if abs(denom) < 1e-12:
        return float("nan")
    return float(np.sum(zc * (y - y.mean())) / denom)


def figure_experiment_design_map() -> dict[str, Any]:
    fig_id = "fig_experiment_design_map"
    rows = [
        {
            "step": "Queue\nwalk",
            "claim": "visible\nsupport",
            "gate": "Fig. 2",
        },
        {
            "step": "Residual\noutcome",
            "claim": r"$\Delta P_T-M$",
            "gate": "Figs. 2, 3",
        },
        {
            "step": "Proxy-error\ngate",
            "claim": r"$\mathrm{Cov}(Z,e\mid B)$",
            "gate": "Fig. 4",
        },
        {
            "step": "Residual-CIV\ngate",
            "claim": "valid\ngraph",
            "gate": "Fig. 6",
        },
        {
            "step": "Natural-IV\nscreen",
            "claim": "no promoted\nendogenous $Z$",
            "gate": "Table 3",
        },
        {
            "step": "External\nboundary",
            "claim": "mechanics\nonly",
            "gate": "Fig. 5",
        },
    ]
    df = pd.DataFrame(rows)
    source = write_source(fig_id, df)

    fig, ax = plt.subplots(figsize=(8.4, 2.35), constrained_layout=True)
    ax.set_axis_off()
    ax.text(
        0.01,
        0.98,
        "Evidence flow",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
    )
    left = 0.015
    right = 0.985
    top = 0.80
    box_h = 0.34
    gate_y = 0.20
    gap = 0.018
    n = len(df)
    box_w = (right - left - gap * (n - 1)) / n
    for i, row in df.iterrows():
        x = left + i * (box_w + gap)
        ax.add_patch(
            Rectangle(
                (x, top - box_h),
                box_w,
                box_h,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor=COLORS["black"],
                linewidth=0.8,
            )
        )
        ax.text(x + box_w / 2, top - 0.085, f"{i + 1}. {row['step']}", transform=ax.transAxes, ha="center", va="center", fontsize=7.3, fontweight="bold", linespacing=0.95)
        ax.text(x + box_w / 2, top - 0.225, row["claim"], transform=ax.transAxes, ha="center", va="center", fontsize=7.2, linespacing=0.95)
        if row["gate"]:
            ax.text(x + box_w / 2, gate_y, row["gate"], transform=ax.transAxes, ha="center", va="center", fontsize=6.1, color=COLORS["gray"], linespacing=0.95)
        if i < n - 1:
            ax.text(x + box_w + gap / 2, top - box_h / 2, "→", transform=ax.transAxes, ha="center", va="center", fontsize=10.0, color=COLORS["black"])

    caption = "Evidence-flow diagram: displayed mechanics first, residual outcome second, proxy-error and graph-validity gates before natural-IV screening, with external L2/L3 mechanics as a boundary audit only."
    return _save(
        fig,
        fig_id=fig_id,
        title="Evidence flow",
        claim=caption,
        design_ids=["mechanics_block", "proxy_error_block", "residual_civ_block", "native_schedule_block", "external_lob_expanded_audit"],
        source_artifacts=["experiments/setup/full_experiment_designs.yaml", "quality_revision/claim_to_figure_map.yaml"],
        source_data=source,
        evidence_type="design_map",
        caption=caption,
        extra_metadata={"sample_size": int(len(df)), "clusters_or_cells": int(len(df)), "inference_method": "design taxonomy"},
    )


def figure_oracle_decomposition_atlas() -> dict[str, Any]:
    fig_id = "fig_oracle_decomposition_atlas"
    branch = read_csv("data/market_impact_study/strategic_groundtruth/extended_intervention_ladder/latest/data/one_shot_branch_rows.csv")
    branch = branch[branch["horizon"].eq(150)].copy()
    branch = branch[branch["liquidity_scale"].isin([1.5, 3.5, 10.0])]
    branch = branch[branch["q_economic_intended"].between(-260, 260)]
    branch["terminal_total"] = branch["impact_by_horizon"]
    branch["displayed_mechanics"] = branch["f_walk"]
    branch["propagation_residual"] = branch["propagation_residual_by_horizon"]
    summaries: list[pd.DataFrame] = []
    for outcome in ["terminal_total", "displayed_mechanics", "propagation_residual"]:
        s = mean_ci(branch, ["liquidity_scale", "q_economic_intended"], outcome)
        s["outcome"] = outcome
        s["support_rate"] = branch.groupby(["liquidity_scale", "q_economic_intended"], observed=True)["support_ok"].mean().to_numpy()
        s["clipping_rate"] = branch.groupby(["liquidity_scale", "q_economic_intended"], observed=True)["clipped_flag"].mean().to_numpy()
        summaries.append(s)
    source_rows = pd.concat(summaries, ignore_index=True)

    curve = read_csv("data/market_impact_study/strategic_groundtruth/extended_intervention_ladder/component_proxy_curves/run_settings_curve_points.csv")
    fit = read_csv("data/market_impact_study/strategic_groundtruth/extended_intervention_ladder/component_proxy_curves/run_settings_fit_summary.csv")
    curve = curve[
        curve["setting_id"].eq("overnight_20260515")
        & curve["outcome"].eq("mechanics")
        & curve["horizon"].eq(0)
        & curve["proxy"].isin(["x_lots", "x_over_full_depth"])
    ].copy()
    fit = fit[
        fit["setting_id"].eq("overnight_20260515")
        & fit["outcome"].eq("mechanics")
        & fit["horizon"].eq(0)
        & fit["proxy"].isin(["x_lots", "x_over_full_depth"])
    ][["proxy", "r2", "beta"]].copy()
    source = write_source(fig_id, pd.concat([source_rows.assign(panel="decomposition"), curve.assign(panel="normalizer_curve"), fit.assign(panel="normalizer_fit")], ignore_index=True, sort=False))

    diagnostics = (
        branch.groupby("liquidity_scale", observed=True)
        .agg(
            min_support=("support_ok", "mean"),
            max_clipping=("clipped_flag", "mean"),
            max_abs_walk_error=("walk_error", lambda s: float(np.max(np.abs(s)))),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 3.28), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.78, wspace=0.16)
    outcomes = [
        ("terminal_total", "Terminal total", COLORS["blue"], "-", "o", 1.95, 0.11),
        ("displayed_mechanics", "Displayed mechanics", COLORS["orange"], (0, (4, 1.8)), "s", 1.75, 0.00),
        ("propagation_residual", "Residual propagation", COLORS["purple"], "-", "D", 1.70, 0.10),
    ]
    liquidities = [1.5, 3.5, 10.0]
    handles: dict[str, Any] = {}
    for idx, (ax, liq) in enumerate(zip(axes, liquidities)):
        row_values = source_rows[source_rows["liquidity_scale"].eq(liq)]
        y_min = float(row_values["ci_low"].min())
        y_max = float(row_values["ci_high"].max())
        pad = 0.08 * max(y_max - y_min, 1.0)
        ax.axhline(0, color=COLORS["black"], lw=0.80, alpha=0.58)
        ax.axvline(0, color=COLORS["black"], lw=0.75, alpha=0.45, linestyle=(0, (2.4, 2.4)))

        bad = row_values[(row_values["support_rate"] < 0.999) | (row_values["clipping_rate"] > 0)]
        for q in sorted(bad["q_economic_intended"].dropna().unique()):
            ax.axvspan(float(q) - 11, float(q) + 11, color="#f3f4f6", zorder=0)

        for outcome, label, color, linestyle, marker, linewidth, band_alpha in outcomes:
            sub = (
                source_rows[
                    (source_rows["liquidity_scale"].eq(liq)) & (source_rows["outcome"].eq(outcome))
                ]
                .sort_values("q_economic_intended")
                .copy()
            )
            x = sub["q_economic_intended"].to_numpy(dtype=float)
            y = sub["mean"].to_numpy(dtype=float)
            lo = sub["ci_low"].to_numpy(dtype=float)
            hi = sub["ci_high"].to_numpy(dtype=float)
            if band_alpha > 0:
                ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, linewidth=0, zorder=1)
            (line,) = ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                lw=linewidth,
                marker=marker,
                markersize=3.4,
                markeredgewidth=0.0,
                zorder=3,
                label=label,
            )
            handles[label] = line

        diag = diagnostics[diagnostics["liquidity_scale"].eq(liq)].iloc[0]
        support_pct = 100.0 * float(diag["min_support"])
        walk_error = float(diag["max_abs_walk_error"])
        walk_text = "0" if walk_error < 5e-12 else f"{walk_error:.2g}"
        ax.text(
            0.04,
            0.06,
            f"support {support_pct:.0f}%\nmax |walk err| {walk_text}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color=COLORS["black"],
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "linewidth": 0.4, "pad": 2.0, "alpha": 0.88},
        )
        ax.set_title(f"{chr(ord('A') + idx)}. Liquidity {liq:g}", fontsize=10.5, fontweight="bold", pad=6)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_xlim(-280, 280)
        ax.set_xlabel("economic signed flow Q")
        if idx == 0:
            ax.set_ylabel("response (ticks)")
        else:
            ax.set_yticklabels([])
        ax.set_xticks([-260, -160, -60, 0, 60, 160, 260])
        ax.grid(True, axis="both", color=COLORS["light_gray"], alpha=0.58)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    legend_order = ["Terminal total", "Displayed mechanics", "Residual propagation"]
    fig.legend(
        [handles[label] for label in legend_order],
        legend_order,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=3,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.4,
    )

    caption = "Cloned oracle atlas decomposes terminal response into displayed mechanics and residual propagation across liquidity settings; normalizer diagnostics are retained in metadata and caption rather than plotted as an explanatory text panel."
    return _save(
        fig,
        fig_id=fig_id,
        title="Oracle mechanics-propagation decomposition atlas",
        claim=caption,
        design_ids=["mechanics_block"],
        source_artifacts=[
            "data/market_impact_study/strategic_groundtruth/extended_intervention_ladder/latest/data/one_shot_branch_rows.csv",
            "data/market_impact_study/strategic_groundtruth/extended_intervention_ladder/component_proxy_curves/run_settings_fit_summary.csv",
        ],
        source_data=source,
        evidence_type="cloned_simulator_oracle",
        caption=caption,
        extra_metadata={
            "sample_size": int(len(branch)),
            "normalizer_r2_raw": float(fit[fit["proxy"].eq("x_lots")]["r2"].iloc[0]),
            "normalizer_r2_depth": float(fit[fit["proxy"].eq("x_over_full_depth")]["r2"].iloc[0]),
            "max_abs_walk_error": float(branch["walk_error"].abs().max()),
            "max_clipping_rate": float(branch["clipped_flag"].mean()),
            "min_support_rate": float(branch["support_ok"].mean()),
        },
    )


def figure_proxy_error_covariance_identity() -> dict[str, Any]:
    fig_id = "fig_proxy_error_covariance_identity"
    sweep = read_csv("artifacts/proxy_error_block/proxy_error_sweep.csv")
    sweep = sweep.copy()
    sweep["proxy_bias"] = sweep["proxy_residual_iv"] - sweep["exact_residual_iv"]
    sweep["proxy_bias_ci_low"] = sweep["proxy_ci_low"] - sweep["exact_residual_iv"]
    sweep["proxy_bias_ci_high"] = sweep["proxy_ci_high"] - sweep["exact_residual_iv"]
    sweep["abs_prediction_gap"] = (sweep["predicted_target_shift"] - sweep["observed_target_shift"]).abs()
    source = write_source(fig_id, sweep.assign(panel="proxy_bias_sweep"))

    fig = plt.figure(figsize=(7.35, 4.15), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.02, 1.18], wspace=0.16)
    ax_identity = fig.add_subplot(gs[0, 0])
    ax_rep = fig.add_subplot(gs[0, 1])

    families = [
        ("linear_depth_proxy", "linear depth", COLORS["blue"], "o"),
        ("square_root_proxy", "square-root", COLORS["orange"], "^"),
        ("sample_split_learned_proxy", "sample-split learned", COLORS["purple"], "s"),
    ]
    max_gap = float(sweep["abs_prediction_gap"].max())
    max_shift = float(sweep["proxy_bias"].abs().max())
    n_cells = int(sweep[["proxy_family", "instrument_proxy_error_correlation"]].drop_duplicates().shape[0])
    min_first_stage = float(sweep["first_stage_F"].min())

    # A. Direct check of the covariance identity in Eq. (mechanical_error_bias).
    identity_min = float(min(sweep["predicted_target_shift"].min(), sweep["observed_target_shift"].min()))
    identity_max = float(max(sweep["predicted_target_shift"].max(), sweep["observed_target_shift"].max()))
    pad = 0.055 * (identity_max - identity_min)
    ax_identity.plot(
        [identity_min - pad, identity_max + pad],
        [identity_min - pad, identity_max + pad],
        color=COLORS["black"],
        lw=1.0,
        label="45-degree line",
        zorder=1,
    )
    for key, label, color, marker in families:
        sub = sweep[sweep["proxy_family"].eq(key)]
        rmse = float(sub["mechanical_proxy_rmse"].mean())
        ax_identity.scatter(
            sub["predicted_target_shift"],
            sub["observed_target_shift"],
            s=24,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.45,
            label=f"{label}; RMSE {rmse:.2f}",
            zorder=2,
        )
    ax_identity.axhline(0, color=COLORS["gray"], lw=0.7)
    ax_identity.axvline(0, color=COLORS["gray"], lw=0.7)
    ax_identity.set_xlim(identity_min - pad, identity_max + pad)
    ax_identity.set_ylim(identity_min - pad, identity_max + pad)
    ax_identity.set_xlabel(r"predicted covariance shift")
    ax_identity.set_ylabel(r"observed IV bias")
    ax_identity.set_title("A. Covariance term explains bias", fontweight="bold")
    ax_identity.text(
        0.03,
        0.97,
        f"max gap {max_gap:.1e}",
        transform=ax_identity.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": COLORS["light_gray"], "pad": 2.0},
    )

    # B. Representative rows: the same proxy is harmless only when its error is
    # instrument-orthogonal.
    corr_specs = [
        (-0.9, "negative corr.", COLORS["orange"], "v", -0.22),
        (0.0, "orthogonal", COLORS["black"], "o", 0.0),
        (0.9, "positive corr.", COLORS["blue"], "^", 0.22),
    ]
    y_base = {key: len(families) - 1 - i for i, (key, *_rest) in enumerate(families)}
    y_labels = []
    y_ticks = []
    for key, label, _color, _marker in families:
        rmse = sweep.loc[sweep["proxy_family"].eq(key), "mechanical_proxy_rmse"].mean()
        y_ticks.append(y_base[key])
        y_labels.append(f"{label}\nRMSE {rmse:.2f}")

    for corr, label, color, marker, offset in corr_specs:
        sub = sweep[np.isclose(sweep["instrument_proxy_error_correlation"], corr)].copy()
        sub["y"] = sub["proxy_family"].map(y_base) + offset
        ax_rep.errorbar(
            sub["proxy_bias"],
            sub["y"],
            xerr=[
                sub["proxy_bias"] - sub["proxy_bias_ci_low"],
                sub["proxy_bias_ci_high"] - sub["proxy_bias"],
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            markersize=4.6,
            elinewidth=1.0,
            capsize=2.0,
            linestyle="none",
            label=label,
            zorder=3,
        )
    ax_rep.axvline(0, color=COLORS["black"], lw=0.9)
    ax_rep.set_yticks(y_ticks)
    ax_rep.set_yticklabels(y_labels)
    ax_rep.set_xlabel(r"$\hat\beta(\widehat M)-\hat\beta(M)$")
    ax_rep.set_title("B. Same proxy, different error correlation", fontweight="bold")
    ax_rep.set_ylim(-0.55, len(families) - 0.45)
    bias_limit = max(0.08, float(np.nanmax(np.abs(np.r_[sweep["proxy_bias_ci_low"], sweep["proxy_bias_ci_high"]]))))
    ax_rep.set_xlim(-1.13 * bias_limit, 1.13 * bias_limit)
    ax_rep.grid(axis="x", color=COLORS["light_gray"], alpha=0.72)
    ax_rep.grid(axis="y", color=COLORS["light_gray"], alpha=0.25)

    handles, labels = ax_identity.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=3,
        fontsize=7.2,
        handlelength=1.5,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    ax_rep.legend(
        frameon=False,
        loc="lower right",
        fontsize=7.2,
        handlelength=1.2,
        borderaxespad=0.2,
    )

    caption = "Proxy-error covariance explains residual-IV bias: across three proxy mechanics functions and controlled instrument-error correlations, observed proxy-induced bias tracks the covariance-shift term in the mechanical-error identity. Bias is near zero only when proxy error is instrument-orthogonal; proxy RMSE affects magnitude but not validity by itself."
    return _save(
        fig,
        fig_id=fig_id,
        title="Proxy-error covariance explains residual-IV bias",
        claim=caption,
        design_ids=["proxy_error_block"],
        source_artifacts=[
            "artifacts/proxy_error_block/proxy_error_sweep.csv",
        ],
        source_data=source,
        evidence_type="generated_proxy_function_bias_diagnostic",
        caption=caption,
        extra_metadata={
            "max_abs_shift_prediction_gap": max_gap,
            "max_abs_observed_shift": max_shift,
            "proxy_family_by_correlation_cells": n_cells,
            "min_first_stage_f": min_first_stage,
        },
    )


def figure_parent_schedule_bundle() -> dict[str, Any]:
    fig_id = "fig_parent_schedule_bundle"
    path = read_csv("data/figure_source/main/fig_metaorder_slice_decomposition.csv")
    matched = read_csv("artifacts/quality/matched_mechanics_path_dependence/results.csv")
    shape = read_csv("data/market_impact_study/native_stress_tests/native_metaorder_shape/results.csv")

    shape_summary = (
        shape.groupby("K", observed=True)[["sequential_mechanics", "terminal_response"]]
        .agg(lambda s: float(np.mean(np.abs(s))))
        .reset_index()
    )
    retention_summary = (
        shape.assign(
            abs_terminal=lambda d: d["terminal_response"].abs(),
            abs_mechanics=lambda d: d["sequential_mechanics"].abs(),
        )
        .groupby("K", observed=True)[["abs_terminal", "abs_mechanics"]]
        .sum()
        .reset_index()
    )
    retention_summary["retention_ratio"] = retention_summary["abs_terminal"] / retention_summary["abs_mechanics"].replace(0, np.nan)
    source = write_source(
        fig_id,
        pd.concat(
            [
                path.assign(panel="fixed_parent_path"),
                matched.assign(panel="matched_schedule_path"),
                shape_summary.assign(panel="slicing_depth_components"),
                retention_summary.assign(panel="terminal_retention"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )

    fig = plt.figure(figsize=(7.35, 4.95), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[0.88, 1.18], width_ratios=[1.55, 0.92], hspace=0.10, wspace=0.28)

    ax_path = fig.add_subplot(gs[0, :])
    slice_grid = [1, 2, 5, 10]
    red_palette = {1: "#7f2704", 2: "#b35806", 5: COLORS["red"], 10: "#f46d43"}
    for k in slice_grid:
        sub = path[path["K"].eq(k)].sort_values("relative_time")
        color = red_palette[k]
        ax_path.plot(sub["relative_time"], sub["mean"], color=color, linewidth=1.45, label=f"K={k}")
        child_times = [float(value) for value in str(sub["child_times_relative"].dropna().iloc[0]).split(";") if value]
        ax_path.scatter(
            child_times,
            np.interp(child_times, sub["relative_time"].to_numpy(float), sub["mean"].to_numpy(float)),
            s=8,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
    ax_path.axvspan(0, 9, color="#f3f4f6", alpha=0.95, zorder=-1)
    ax_path.axhline(0, color=COLORS["black"], linewidth=0.75)
    ax_path.set_xlim(0, 120)
    ax_path.set_ylim(-0.2, 4.65)
    ax_path.set_title("A. Fixed parent path changes with slicing", fontweight="bold", fontsize=10.0)
    ax_path.set_xlabel("time since first child (events)")
    ax_path.set_ylabel("do(0)-subtracted ticks")
    ax_path.legend(frameon=False, ncol=4, loc="upper right", fontsize=7.1, handlelength=1.4, columnspacing=0.9)
    ax_path.text(
        0.98,
        0.16,
        "shaded: execution window\n"
        "dots: child orders",
        transform=ax_path.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["gray"],
    )
    ax_path.grid(True, color="#e5e7eb", linewidth=0.5)

    ax_shift = fig.add_subplot(gs[1, 0])
    order = ["one_shot", "front_loaded", "burst_pause_burst", "twap_2", "randomized_jitter", "twap_5", "twap_10", "twap_20", "back_loaded"]
    arm_labels = {
        "one_shot": "one-shot",
        "front_loaded": "front loaded",
        "burst_pause_burst": "burst-pause",
        "twap_2": "TWAP 2",
        "randomized_jitter": "jitter",
        "twap_5": "TWAP 5",
        "twap_10": "TWAP 10",
        "twap_20": "TWAP 20",
        "back_loaded": "back loaded",
    }
    sub = matched.set_index("arm").loc[order].reset_index()
    y = np.arange(len(sub))[::-1]
    max_balance = float(matched["abs_fseq_balance_ticks"].max())
    balance_band = max(0.08, 8.0 * max_balance)
    ax_shift.axvspan(-balance_band, balance_band, color=COLORS["light_gray"], alpha=0.5, zorder=0)
    ax_shift.axvline(0, color=COLORS["black"], lw=0.8, zorder=1)
    for i, row in sub.iterrows():
        estimate = float(row["estimate"])
        color = COLORS["blue"] if estimate > 0.05 else (COLORS["orange"] if estimate < -0.05 else COLORS["black"])
        ax_shift.errorbar(
            estimate,
            y[i],
            xerr=[[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.0,
            elinewidth=1.0,
            ms=3.6,
            zorder=3,
        )
        label_x = float(row["ci_high"]) + 0.08
        ax_shift.text(
            label_x,
            y[i],
            f"{estimate:+.2f}",
            ha="left",
            va="center",
            fontsize=6.6,
            color=COLORS["black"],
        )
    ax_shift.set_yticks(y)
    ax_shift.set_yticklabels([arm_labels[a] for a in sub["arm"]])
    ax_shift.set_xlabel("residual propagation after matched mechanics (ticks)")
    ax_shift.set_title("B. Matched mechanics, different paths", fontweight="bold", fontsize=10.0)
    ax_shift.set_xlim(float(sub["ci_low"].min()) - 0.45, float(sub["ci_high"].max()) + 0.55)
    ax_shift.grid(axis="x", color=COLORS["light_gray"], alpha=0.72)
    ax_shift.grid(axis="y", color=COLORS["light_gray"], alpha=0.25)

    ax_ret = fig.add_subplot(gs[1, 1])
    bar_data = shape_summary.merge(retention_summary[["K", "retention_ratio"]], on="K", how="left").sort_values("K").reset_index(drop=True)
    x = np.arange(len(bar_data))
    width = 0.34
    ax_ret.bar(x - width / 2, bar_data["sequential_mechanics"], width=width, color=COLORS["blue"], alpha=0.88, label=r"$|F_{seq}|$")
    ax_ret.bar(x + width / 2, bar_data["terminal_response"], width=width, color=COLORS["orange"], alpha=0.90, label="terminal")
    for i, row in bar_data.iterrows():
        y_pos = max(float(row["sequential_mechanics"]), float(row["terminal_response"])) + 0.08
        ax_ret.text(x[i], y_pos, f"{float(row['retention_ratio']):.2f}x", ha="center", va="bottom", fontsize=6.8)
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels([f"K={int(k)}" for k in bar_data["K"]])
    ax_ret.set_xlabel("slicing depth")
    ax_ret.set_ylabel("mean abs. ticks")
    ax_ret.set_title("C. Terminal retention", fontweight="bold", fontsize=10.0)
    ax_ret.set_ylim(0.0, float(bar_data[["sequential_mechanics", "terminal_response"]].to_numpy().max()) + 0.45)
    ax_ret.legend(frameon=False, loc="upper left", fontsize=6.8, handlelength=1.3)
    ax_ret.grid(axis="y", color=COLORS["light_gray"], alpha=0.65)

    caption = (
        "Parent-schedule diagnostics: fixed-size cloned parent paths vary with slicing, residual propagation remains path dependent "
        f"after sequential displayed mechanics are matched (max |F_seq| imbalance {max_balance:.3f} ticks), and native slicing shows "
        "that child-walk mechanics do not survive one-for-one to the terminal horizon."
    )
    return _save(
        fig,
        fig_id=fig_id,
        title="Parent schedule diagnostics",
        claim=caption,
        design_ids=["metaorder_block", "native_factorial_metaorder_refill"],
        source_artifacts=[
            "data/figure_source/main/fig_metaorder_slice_decomposition.csv",
            "artifacts/quality/matched_mechanics_path_dependence/results.csv",
            "data/market_impact_study/native_stress_tests/native_metaorder_shape/results.csv",
        ],
        source_data=source,
        evidence_type="stored_and_generated_parent_schedule_diagnostics",
        caption=caption,
        extra_metadata={
            "max_abs_fseq_balance_ticks": float(matched["abs_fseq_balance_ticks"].max()),
            "min_support_rate": float(matched["support_rate"].min()),
            "max_clipping_rate": float(matched["clipping_rate"].max()),
        },
    )


def figure_schedule_path_decomposition() -> dict[str, Any]:
    fig_id = "fig_schedule_path_decomposition"
    matched = read_csv("artifacts/quality/matched_mechanics_path_dependence/results.csv")
    resilience = read_csv("artifacts/quality/resilience_recovery_kernel/results.csv")
    shape = read_csv("data/market_impact_study/native_stress_tests/native_metaorder_shape/results.csv")

    shape_summary = (
        shape.groupby("K", observed=True)[["sequential_mechanics", "between_slice_propagation", "terminal_response"]]
        .agg(lambda s: float(np.mean(np.abs(s))))
        .reset_index()
    )
    retention_summary = (
        shape.assign(
            abs_terminal=lambda d: d["terminal_response"].abs(),
            abs_mechanics=lambda d: d["sequential_mechanics"].abs(),
        )
        .groupby("K", observed=True)[["abs_terminal", "abs_mechanics"]]
        .sum()
        .reset_index()
    )
    retention_summary["retention_ratio"] = retention_summary["abs_terminal"] / retention_summary["abs_mechanics"].replace(0, np.nan)
    source = write_source(
        fig_id,
        pd.concat(
            [
                matched.assign(panel="matched_schedule_path"),
                resilience.assign(panel="resilience_kernel"),
                shape_summary.assign(panel="slicing_depth_components"),
                retention_summary.assign(panel="terminal_retention"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )

    fig = plt.figure(figsize=(8.4, 4.65), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.55, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    order = ["one_shot", "front_loaded", "burst_pause_burst", "twap_2", "randomized_jitter", "twap_5", "twap_10", "twap_20", "back_loaded"]
    arm_labels = {
        "one_shot": "one-shot block",
        "front_loaded": "front loaded",
        "burst_pause_burst": "burst-pause-burst",
        "twap_2": "TWAP 2",
        "randomized_jitter": "randomized jitter",
        "twap_5": "TWAP 5 reference",
        "twap_10": "TWAP 10",
        "twap_20": "TWAP 20",
        "back_loaded": "back loaded",
    }
    sub = matched.set_index("arm").loc[order].reset_index()
    y = np.arange(len(sub))[::-1]
    max_balance = float(matched["abs_fseq_balance_ticks"].max())
    support_rate = float(matched["support_rate"].min())
    clipping_rate = float(matched["clipping_rate"].max())
    balance_band = max(0.08, 8.0 * max_balance)
    ax.axvspan(-balance_band, balance_band, color=COLORS["light_gray"], alpha=0.52, zorder=0)
    ax.axvline(0, color=COLORS["black"], lw=0.85, zorder=1)
    for i, row in sub.iterrows():
        estimate = float(row["estimate"])
        color = COLORS["blue"] if estimate > 0.05 else (COLORS["orange"] if estimate < -0.05 else COLORS["black"])
        ax.errorbar(
            estimate,
            y[i],
            xerr=[[estimate - float(row["ci_low"])], [float(row["ci_high"]) - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.2,
            elinewidth=1.25,
            ms=4.2,
            zorder=3,
        )
        label_x = float(row["ci_high"]) + 0.08 if estimate >= 0 else float(row["ci_low"]) - 0.08
        ax.text(
            label_x,
            y[i],
            f"{estimate:+.2f}",
            ha="left" if estimate >= 0 else "right",
            va="center",
            fontsize=7.0,
            color=COLORS["black"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([arm_labels[a] for a in sub["arm"]])
    ax.set_xlabel("residual propagation after matched mechanics (ticks)")
    ax.set_title(r"A. Mechanics matched; residual path still moves", fontweight="bold", fontsize=11.5)
    ax.set_xlim(float(sub["ci_low"].min()) - 0.45, float(sub["ci_high"].max()) + 0.55)
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.70)
    ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.28)
    ax.text(
        0.02,
        0.97,
        rf"common support; max $|F_{{seq}}$ gap = {max_balance:.3f} ticks"
        "\n"
        rf"support {support_rate:.0%}; clip {clipping_rate:.0%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": COLORS["light_gray"], "pad": 2.2, "alpha": 0.90},
    )

    ax = fig.add_subplot(gs[0, 1])
    bar_data = shape_summary.merge(retention_summary[["K", "retention_ratio"]], on="K", how="left").sort_values("K").reset_index(drop=True)
    x = np.arange(len(bar_data))
    width = 0.34
    ax.bar(
        x - width / 2,
        bar_data["sequential_mechanics"],
        width=width,
        color=COLORS["blue"],
        alpha=0.88,
        label=r"mean $|F_{seq}|$",
    )
    ax.bar(
        x + width / 2,
        bar_data["terminal_response"],
        width=width,
        color=COLORS["orange"],
        alpha=0.90,
        label="mean terminal |response|",
    )
    for i, row in bar_data.iterrows():
        x_pos = x[i]
        y_pos = max(float(row["sequential_mechanics"]), float(row["terminal_response"])) + 0.08
        ax.text(
            x_pos,
            y_pos,
            f"retained\n{float(row['retention_ratio']):.2f}x",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color=COLORS["black"],
            linespacing=0.95,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={int(k)}" for k in bar_data["K"]])
    ax.set_xlabel("slicing depth K")
    ax.set_ylabel("mean absolute ticks")
    ax.set_title("B. Child-walk mechanics do not\nsurvive one-for-one", fontweight="bold", fontsize=11.5)
    ax.set_ylim(0.0, float(bar_data[["sequential_mechanics", "terminal_response"]].to_numpy().max()) + 0.45)
    ax.legend(frameon=False, loc="upper left", fontsize=7.2, handlelength=1.6)
    ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.62)

    caption = "Schedule-path diagnostics show that residual propagation remains path dependent even after sequential displayed mechanics are matched; native slicing also shows that child-walk mechanics do not survive one-for-one to the terminal horizon."
    return _save(
        fig,
        fig_id=fig_id,
        title="Schedule path matters after matched mechanics",
        claim=caption,
        design_ids=["metaorder_block", "native_factorial_metaorder_refill"],
        source_artifacts=[
            "artifacts/quality/matched_mechanics_path_dependence/results.csv",
            "artifacts/quality/resilience_recovery_kernel/results.csv",
            "data/market_impact_study/native_stress_tests/native_metaorder_shape/results.csv",
        ],
        source_data=source,
        evidence_type="generated_and_native_schedule_diagnostics",
        caption=caption,
    )


def figure_civ_validity_frontier() -> dict[str, Any]:
    fig_id = "fig_civ_validity_frontier"
    focus = read_csv("data/figure_source/main/fig_civ_estimator_strength.csv")
    focus = focus[focus["panel"].eq("focus_estimate")].copy()
    residual = read_csv("artifacts/residual_civ_block/estimator_results.csv")
    frontier = read_csv("data/market_impact_study/native_stress_tests/native_assignment_frontier/results.csv")
    ablation = read_csv("data/market_impact_study/civ_identification/section6_civ_identification/section6_conditioning_ablation.csv")
    mc = read_csv("data/market_impact_study/civ_identification/section6_civ_identification/generated_civ_monte_carlo_summary.csv")

    panel_a_rows = []
    lookup = {row["estimator"]: row for _, row in focus.iterrows()}
    for label, key in [
        ("OLS raw terminal", "OLS"),
        ("naive IV", "Naive IV"),
        ("linear CIV", "Linear CIV"),
        ("flexible cross-fit CIV", "Cross-fit flexible CIV"),
    ]:
        row = lookup[key]
        panel_a_rows.append({"estimator": label, "estimate": row["estimate"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "target": row["oracle"]})
    ols_resid = residual[residual["estimator"].eq("OLS")].iloc[0]
    panel_a_rows.insert(1, {"estimator": "OLS residual", "estimate": ols_resid["estimate"], "ci_low": ols_resid["ci_low"], "ci_high": ols_resid["ci_high"], "target": ols_resid["target"]})
    panel_a = pd.DataFrame(panel_a_rows)

    post = ablation[ablation["conditioning_set"].eq("invalid post-treatment state")].iloc[0]
    panel_b = pd.concat(
        [
            frontier.assign(source="native_assignment_frontier")[["design_label", "iv_estimate", "target", "target_gap", "first_stage_F", "pass_fail", "first_failed_diagnostic", "source"]],
            pd.DataFrame(
                [
                    {
                        "design_label": "post-child controls",
                        "iv_estimate": post["estimate"],
                        "target": post["oracle"],
                        "target_gap": post["bias"],
                        "first_stage_F": post["first_stage_f"],
                        "pass_fail": "fail",
                        "first_failed_diagnostic": "post-treatment controls",
                        "source": "generated_conditioning_ablation",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    panel_b["plot_label"] = panel_b["design_label"].replace(
        {
            "native pre-assignment": "valid randomized Z",
            "weak": "weak Z",
            "descendant": "descendant control",
            "direct contamination": "direct contamination",
            "balance failure": "alpha-correlated natural IV",
            "pre-state adaptive; conditioned": "valid pre-state adaptive",
        }
    )

    panel_c = mc[
        mc["residualizer"].eq("flexible_crossfit")
        & mc["rho_alpha"].eq(0.8)
        & mc["rho_z"].eq(0.62)
        & mc["n_rows"].eq(3000)
    ].copy()
    source_df = pd.concat(
        [
            panel_a.assign(panel="estimator_comparison"),
            panel_b.assign(panel="invalid_design_ablation"),
            panel_c.assign(panel="mc500"),
        ],
        ignore_index=True,
        sort=False,
    )

    rows: list[dict[str, Any]] = []

    def add_gap_row(label: str, estimate: float, target_value: float, ci_low: float | None, ci_high: float | None, status: str, first_stage: float | None = None) -> None:
        gap = float(estimate - target_value)
        rows.append(
            {
                "label": label,
                "gap": gap,
                "ci_low_gap": None if ci_low is None else float(ci_low - target_value),
                "ci_high_gap": None if ci_high is None else float(ci_high - target_value),
                "status": status,
                "first_stage_F": first_stage,
            }
        )

    residual_lookup = {row["estimator"]: row for _, row in residual.iterrows()}
    focus_lookup = {row["estimator"]: row for _, row in focus[focus["panel"].eq("focus_estimate")].iterrows()}
    for key, label, status in [
        ("exact residual IV", "exact residual IV", "valid"),
        ("graph-valid lag CIV", "graph-valid lag CIV", "valid"),
    ]:
        row = residual_lookup[key]
        add_gap_row(label, row["estimate"], row["target"], row["ci_low"], row["ci_high"], status, row["first_stage_f"])
    row = focus_lookup["Cross-fit flexible CIV"]
    add_gap_row("cross-fit flexible CIV", row["estimate"], row["oracle"], row["ci_low"], row["ci_high"], "valid", row["first_stage_f"])
    row = residual_lookup["weak-IV residual IV"]
    add_gap_row("weak-IV residual IV (unstable)", row["estimate"], row["target"], row["ci_low"], row["ci_high"], "weak", row["first_stage_f"])
    for key, label in [("Naive IV", "naive IV")]:
        row = residual_lookup[key]
        add_gap_row(label, row["estimate"], row["target"], row["ci_low"], row["ci_high"], "invalid", row["first_stage_f"])
    post = ablation[ablation["conditioning_set"].eq("invalid post-treatment state")].iloc[0]
    add_gap_row("post-child controls", post["estimate"], post["oracle"], post["ci_low"], post["ci_high"], "invalid", post["first_stage_f"])
    for design, label in [("descendant control", "descendant control"), ("direct contamination", "direct contamination")]:
        row = panel_b[panel_b["plot_label"].eq(design)].iloc[0]
        add_gap_row(label, row["iv_estimate"], row["target"], None, None, "invalid", row["first_stage_F"])

    plot_df = pd.DataFrame(rows)
    display_labels = {
        "exact residual IV": "exact residual IV",
        "graph-valid lag CIV": "lag CIV",
        "cross-fit flexible CIV": "cross-fit CIV",
        "weak-IV residual IV (unstable)": "weak residual IV",
        "naive IV": "naive IV",
        "post-child controls": "post-child controls",
        "descendant control": "descendant control",
        "direct contamination": "direct contamination",
    }
    plot_df["display_label"] = plot_df["label"].map(display_labels).fillna(plot_df["label"])
    xmax = 0.95
    xmin = -0.55
    plot_df["plot_gap"] = plot_df["gap"].clip(lower=xmin + 0.03, upper=xmax - 0.03)
    plot_df["offscale"] = (plot_df["gap"] != plot_df["plot_gap"])
    source = write_source(fig_id, pd.concat([source_df, plot_df.assign(panel="main_gap_forest")], ignore_index=True, sort=False))

    fig, ax = plt.subplots(figsize=(5.7, 3.35), constrained_layout=True)
    y = np.arange(len(plot_df))[::-1]
    ax.axvline(0, color=COLORS["black"], lw=0.85)
    ax.axhline(4.5, color=COLORS["light_gray"], lw=0.85)
    ax.axhline(3.5, color=COLORS["light_gray"], lw=0.85)
    ax.text(xmin + 0.02, 7.36, "graph-valid", ha="left", va="center", fontsize=6.4, color=COLORS["gray"])
    ax.text(xmin + 0.02, 4.30, "weak denominator", ha="left", va="center", fontsize=6.4, color=COLORS["gray"])
    ax.text(xmin + 0.02, 3.30, "invalid shortcuts", ha="left", va="center", fontsize=6.4, color=COLORS["gray"])
    style = {
        "valid": (COLORS["blue"], "o"),
        "weak": (COLORS["gray"], "D"),
        "invalid": (COLORS["orange"], "s"),
    }
    for i, row in plot_df.iterrows():
        color, marker = style[str(row["status"])]
        yi = y[i]
        if pd.notna(row["ci_low_gap"]) and not row["offscale"]:
            lo = float(row["ci_low_gap"])
            hi = float(row["ci_high_gap"])
            ax.errorbar(float(row["plot_gap"]), yi, xerr=[[float(row["plot_gap"]) - lo], [hi - float(row["plot_gap"])]], fmt=marker, color=color, ecolor=color, capsize=1.8, elinewidth=0.9, ms=3.5)
        else:
            marker_use = ">" if row["gap"] > xmax else ("<" if row["gap"] < xmin else marker)
            ax.scatter(float(row["plot_gap"]), yi, color=color, marker=marker_use, s=26, zorder=3)
        if row["offscale"]:
            ha = "right" if row["gap"] > xmax else "left"
            dx = -0.035 if row["gap"] > xmax else 0.035
            ax.text(float(row["plot_gap"]) + dx, yi, "off-scale", ha=ha, va="center", fontsize=6.4, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["display_label"], fontsize=7.2)
    ax.set_ylim(-0.55, len(plot_df) - 0.02)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("estimate - declared target", fontsize=7.8)
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_title("Residual-CIV target gap", fontweight="bold", fontsize=9.0, pad=3)
    ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.6)
    target = float(focus_lookup["Cross-fit flexible CIV"]["oracle"])

    caption = "Residual-CIV validity-frontier diagnostics: graph-valid rows match declared scalar or projection targets, while weak denominators, post-treatment controls, descendants, and direct contamination define failure rows."
    return _save(
        fig,
        fig_id=fig_id,
        title="Residual-CIV validity frontier",
        claim=caption,
        design_ids=["residual_civ_block", "native_schedule_block"],
        source_artifacts=[
            "data/figure_source/main/fig_civ_estimator_strength.csv",
            "artifacts/residual_civ_block/estimator_results.csv",
            "data/market_impact_study/native_stress_tests/native_assignment_frontier/results.csv",
            "data/market_impact_study/civ_identification/section6_civ_identification/generated_civ_monte_carlo_summary.csv",
        ],
        source_data=source,
        evidence_type="generated_and_native_civ_diagnostics",
        caption=caption,
        extra_metadata={"mc500_rows_used": int(len(panel_c)), "oracle_target": target},
    )


def figure_native_schedule_target_heatmap() -> dict[str, Any]:
    fig_id = "fig_native_schedule_target_heatmap"
    weights = read_csv("artifacts/additional/civ_target_weight_decomposition/results.csv")
    native = read_csv("artifacts/native_schedule_block/native_schedule_results.csv")
    summary = read_json("artifacts/additional/civ_target_weight_decomposition/summary.json")
    wide = native[native["panel_name"].eq("Wide robustness panel")].iloc[0]

    weights = weights.copy()
    weights["global_signed_weight"] = weights["first_stage_weight"] / float(weights["first_stage_weight"].sum())
    weights["positive_weight"] = weights["first_stage_weight"].clip(lower=0.0)
    weights["positive_weight_share"] = weights["positive_weight"] / float(weights["positive_weight"].sum())
    weights["weighted_contribution"] = weights["global_signed_weight"] * weights["cell_specific_branch_slope"]

    native_plot = native.copy()
    native_plot["target_gap"] = native_plot["estimate"] - native_plot["target"]
    native_plot["ci_low_gap"] = native_plot["ci_low"] - native_plot["target"]
    native_plot["ci_high_gap"] = native_plot["ci_high"] - native_plot["target"]
    panel_labels = {
        "Stored policy-perturbation panel": "Stored schedule",
        "Integrated alpha-policy panel": "Integrated alpha",
        "Expanded robustness panel": "Expanded robust.",
        "Wide robustness panel": "Wide robust.",
        "RL/dynamic-alpha robustness panel": "RL-alpha robust.",
        "RL/dynamic-alpha policy panel": "RL-alpha policy",
        "Permutation audit": "Permutation audit",
    }
    native_plot["short_panel"] = native_plot["panel_name"].map(panel_labels).fillna(native_plot["panel_name"])
    source = write_source(
        fig_id,
        pd.concat(
            [
                weights.assign(panel="weight_distribution"),
                native_plot.assign(panel="native_schedule_estimates"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )

    min_f = float(native["first_stage_f"].min())
    max_gap = float(native_plot["target_gap"].abs().max())

    main_native = native_plot[~native_plot["panel_name"].eq("Permutation audit")].copy()
    main_native = main_native.sort_values("target").reset_index(drop=True)

    fig = plt.figure(figsize=(9.4, 5.15), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.1, 0.9], width_ratios=[1.1, 1.0], hspace=0.24, wspace=0.24)

    ax_est = fig.add_subplot(gs[0, :])
    ypos = np.arange(len(main_native))
    ax_est.hlines(ypos, main_native["target"], main_native["estimate"], color=COLORS["light_gray"], lw=2.0, zorder=1)
    ax_est.errorbar(
        main_native["estimate"],
        ypos,
        xerr=[main_native["estimate"] - main_native["ci_low"], main_native["ci_high"] - main_native["estimate"]],
        fmt="o",
        color=COLORS["blue"],
        ecolor=COLORS["blue"],
        elinewidth=1.0,
        capsize=2.2,
        markersize=4.5,
        label="native IV estimate",
        zorder=3,
    )
    ax_est.scatter(main_native["target"], ypos, marker="D", s=30, color=COLORS["black"], label="declared target", zorder=4)
    ax_est.set_yticks(ypos)
    ax_est.set_yticklabels(main_native["short_panel"])
    ax_est.set_xlabel(r"residual propagation slope $\beta$")
    ax_est.set_title("A. Native estimates vs declared targets", fontweight="bold")
    x_min = float(main_native["ci_low"].min()) - 0.015
    x_max = float(main_native["ci_high"].max()) + 0.015
    ax_est.set_xlim(x_min, x_max)
    ax_est.grid(axis="x", color=COLORS["light_gray"], alpha=0.65)
    ax_est.grid(axis="y", visible=False)
    ax_est.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.6)

    ax_gap = fig.add_subplot(gs[1, 0])
    gap_plot = main_native.sort_values("target_gap").copy()
    ypos = np.arange(len(gap_plot))
    gap_milli = 1000.0 * gap_plot["target_gap"].astype(float)
    gap_colors = [COLORS["blue"] if value >= 0 else COLORS["red"] for value in gap_milli]
    ax_gap.barh(ypos, gap_milli, color=gap_colors, alpha=0.88, height=0.55)
    ax_gap.axvline(0.0, color=COLORS["black"], lw=0.9)
    ax_gap.set_yticks(ypos)
    ax_gap.set_yticklabels(gap_plot["short_panel"], fontsize=7.2)
    ax_gap.set_xlabel(r"estimate - covariance-weighted target ($\times 10^{-3}$)")
    ax_gap.set_title("B. Magnified estimate-target gaps", fontweight="bold")
    max_gap_milli = 1000.0 * max_gap
    ax_gap.set_xlim(-max_gap_milli * 1.55, max_gap_milli * 1.55)
    ax_gap.grid(axis="x", color=COLORS["light_gray"], alpha=0.65)
    ax_gap.grid(axis="y", visible=False)
    for yi, value in zip(ypos, gap_milli, strict=True):
        ha = "left" if value >= 0 else "right"
        offset = 0.018 if value >= 0 else -0.018
        ax_gap.text(float(value) + offset, yi, f"{float(value):+.2f}", ha=ha, va="center", fontsize=6.8, color=COLORS["black"])
    ax_gap.text(
        0.02,
        0.96,
        f"6 panels; rows = {int(main_native['rows'].sum()):,}\nmin first-stage F = {min_f:,.0f}\nmax |gap| = {max_gap_milli:.2f} $\\times 10^{{-3}}$",
        transform=ax_gap.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
    )

    ax_w = fig.add_subplot(gs[1, 1])

    finite_weights = weights[np.isfinite(weights["cell_specific_branch_slope"]) & np.isfinite(weights["positive_weight"])].copy()
    finite_weights = finite_weights[finite_weights["positive_weight"] > 0].copy()

    def weighted_quantile(values: np.ndarray, weight_values: np.ndarray, q: float) -> float:
        order = np.argsort(values)
        xs = values[order]
        ws = weight_values[order]
        cdf = np.cumsum(ws) / float(np.sum(ws))
        return float(xs[min(np.searchsorted(cdf, q), len(xs) - 1)])

    slope_values = finite_weights["cell_specific_branch_slope"].to_numpy(dtype=float)
    weight_values = finite_weights["positive_weight"].to_numpy(dtype=float)
    x_lo = weighted_quantile(slope_values, weight_values, 0.005)
    x_hi = weighted_quantile(slope_values, weight_values, 0.995)
    pad = 0.04 * (x_hi - x_lo)
    bins = np.linspace(x_lo - pad, x_hi + pad, 24)
    ax_w.hist(
        slope_values,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color=COLORS["gray"],
        label="unweighted cells",
    )
    ax_w.hist(
        slope_values,
        bins=bins,
        weights=weight_values,
        density=True,
        histtype="stepfilled",
        alpha=0.22,
        color=COLORS["blue"],
        edgecolor=COLORS["blue"],
        linewidth=0.8,
        label="first-stage weighted",
    )
    simple_target = float(summary["unweighted_mean"])
    weighted_target = float(summary["covariance_weighted_target"])
    ax_w.axvline(simple_target, color=COLORS["gray"], lw=1.2, linestyle="--")
    ax_w.axvline(weighted_target, color=COLORS["black"], lw=1.3)
    ax_w.set_xlim(float(bins[0]), float(bins[-1]))
    ax_w.set_xlabel(r"cell-specific branch slope")
    ax_w.set_ylabel("density")
    ax_w.set_title("C. First-stage weights define the target", fontweight="bold")
    ax_w.grid(axis="y", color=COLORS["light_gray"], alpha=0.55)
    ax_w.text(
        0.03,
        0.95,
        f"simple mean = {simple_target:.3f}\nweighted target = {weighted_target:.3f}\ndifference = {float(summary['absolute_difference']):.3f}",
        transform=ax_w.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=COLORS["black"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.0},
    )
    ax_w.legend(loc="upper right", frameon=False, fontsize=7.0)

    caption = "Native randomized schedules recover their own first-stage-covariance weighted residual targets; the simple average branch slope is a different target."
    return _save(
        fig,
        fig_id=fig_id,
        title="Native randomized schedules recover their weighted targets",
        claim=caption,
        design_ids=["native_schedule_block"],
        source_artifacts=[
            "artifacts/additional/civ_target_weight_decomposition/results.csv",
            "artifacts/additional/civ_target_weight_decomposition/summary.json",
            "artifacts/native_schedule_block/native_schedule_results.csv",
        ],
        source_data=source,
        evidence_type="native_event_engine_target_weight_diagnostic",
        caption=caption,
        extra_metadata={
            "weighted_target": float(summary["covariance_weighted_target"]),
            "simple_average": float(summary["unweighted_mean"]),
            "max_native_gap": max_gap,
            "min_first_stage_f": min_f,
        },
    )


def figure_external_lob_boundary() -> dict[str, Any]:
    fig_id = "fig_external_lob_boundary"
    staleness = read_csv("artifacts/external_lob/advanced_l2l3/lobster_staleness_diagnostics.csv")
    proxy = read_csv("artifacts/external_lob/advanced_l2l3/lobster_proxy_model_diagnostics.csv")
    summary = read_json("artifacts/external_lob/advanced_l2l3/advanced_l2l3_summary.json")
    schedule = read_csv_if_exists("artifacts/external_lob/followup_l2l3/lobster_schedule_ranking_summary.csv")
    decision = read_csv_if_exists("artifacts/external_lob/followup_l2l3/lobster_schedule_decision_summary.csv")
    concentration = read_csv_if_exists("artifacts/external_lob/followup_l2l3/coinbase_l3_concentration_summary.csv")
    low_rank = read_csv_if_exists("artifacts/external_lob/advanced_l2l3/lobster_walk_low_rank_errors.csv")
    gate = read_csv_if_exists("artifacts/l2l3_validation/data_contract_gate.csv")

    exact = staleness[staleness["staleness_label"].eq("exact pre-event")].iloc[0]
    one = staleness[staleness["staleness_label"].eq("stale 1 event")].iloc[0]
    ten = staleness[staleness["staleness_label"].eq("stale 10 ms")].iloc[0]
    low = proxy[proxy["model"].eq("low-dimensional book proxy")].iloc[0]
    hidden = proxy[proxy["model"].eq("hidden executions using displayed walk")].iloc[0]
    boundary_rows = [
        {"boundary": "exact pre-event\ndisplayed book", "rmse_ticks": exact["rmse_ticks"], "support_rate": exact["support_rate"], "clipping_rate": 0.0, "hidden_rate": 0.0, "claim": "displayed mechanics"},
        {"boundary": "one-event\nstale book", "rmse_ticks": one["rmse_ticks"], "support_rate": one["support_rate"], "clipping_rate": np.nan, "hidden_rate": 0.0, "claim": "timestamp boundary"},
        {"boundary": "10ms\nstale book", "rmse_ticks": ten["rmse_ticks"], "support_rate": ten["support_rate"], "clipping_rate": np.nan, "hidden_rate": 0.0, "claim": "timestamp boundary"},
        {"boundary": "low-dimensional\nproxy", "rmse_ticks": low["rmse_ticks"], "support_rate": 1.0, "clipping_rate": np.nan, "hidden_rate": 0.0, "claim": "proxy mechanics"},
        {"boundary": "hidden\nexecutions", "rmse_ticks": hidden["rmse_ticks"], "support_rate": 0.9991175432403813, "clipping_rate": 0.0008824567596187787, "hidden_rate": 1.0, "claim": "hidden boundary"},
    ]
    df = pd.concat(
        [
            pd.DataFrame(boundary_rows).assign(panel="boundary_errors"),
            schedule.assign(panel="schedule_ranking"),
            decision.assign(panel="schedule_decision"),
            concentration.assign(panel="l3_concentration"),
            low_rank.assign(panel="walk_svd"),
            gate.assign(panel="claim_gate"),
        ],
        ignore_index=True,
        sort=False,
    )
    source = write_source(fig_id, df)

    rank1 = low_rank[low_rank["rank"].eq(1)].head(1)
    rank1_text = f"{100.0 * float(rank1['residual_variance_share'].iloc[0]):.1f}%" if len(rank1) else "n/a"

    fig = plt.figure(figsize=(8.55, 4.18))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[0.92, 1.08], wspace=0.44, hspace=0.62)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.13)

    ax = fig.add_subplot(gs[0, :])
    proxy_df = pd.DataFrame(boundary_rows).copy()
    labels = ["exact", "1-event stale", "10ms stale", "book proxy", "hidden exec."]
    colors = [COLORS["green"], COLORS["orange"], COLORS["orange"], COLORS["purple"], COLORS["red"]]
    y = np.arange(len(proxy_df))
    ax.barh(y, proxy_df["rmse_ticks"].to_numpy(dtype=float), color=colors, height=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("walk RMSE (ticks)")
    ax.set_title("B. Boundary rows show tick-level error", fontweight="bold")
    x_max = float(proxy_df["rmse_ticks"].max()) * 1.24
    ax.set_xlim(0.0, x_max)
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.62)
    for yi, value in zip(y, proxy_df["rmse_ticks"].to_numpy(dtype=float), strict=True):
        label = "~0" if value < 1e-6 else f"{value:.2f}"
        ax.text(min(value + 0.045 * x_max, 0.96 * x_max), yi, label, ha="left", va="center", fontsize=7.2, color=COLORS["black"])

    ax = fig.add_subplot(gs[1, 0])
    if not schedule.empty:
        show = schedule.sort_values("median_abs_proxy_error_ticks").copy()
        y_pos = np.arange(len(show))
        ax.barh(y_pos, show["median_abs_proxy_error_ticks"], height=0.58, color=COLORS["purple"], alpha=0.84)
        ax.set_yticks(y_pos, [str(item).replace("_", " ") for item in show["schedule_id"]])
        max_error = float(show["median_abs_proxy_error_ticks"].max())
        ax.set_xlim(0.0, max_error * 2.15)
        for yi, value in zip(y_pos, show["median_abs_proxy_error_ticks"].to_numpy(dtype=float), strict=True):
            ax.text(float(value) + 0.035 * max_error, yi, f"{float(value):.2f}", ha="left", va="center", fontsize=6.8)
        if not decision.empty:
            stats = decision.iloc[0]
            ax.text(
                0.98,
                0.08,
                f"starts {int(stats['sampled_parent_starts']):,}\nrank corr {float(stats['median_rank_correlation']):.2f}\ntop-decile miss {100.0 * float(stats['top_decile_disagreement_rate']):.1f}%\np90 regret {float(stats['p90_mechanical_regret_ticks']):.1f} ticks",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.8,
                color=COLORS["gray"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
            )
    ax.set_xlabel("median |scalar proxy - exact| (ticks)")
    ax.set_title("C. Scalar proxy changes schedule rankings", fontweight="bold")
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.55)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_axis_off()
    ax.set_title("D. Claim-tier gate", fontweight="bold", loc="left")

    def gate_any(column: str) -> bool:
        if gate.empty or column not in gate:
            return False
        return bool(gate[column].astype(str).str.lower().isin(["true", "1", "yes"]).any())

    claim_rows = [
        ("displayed depth", "PASS" if gate_any("depth_available") else "GAP"),
        ("visible mechanics", "PASS" if gate_any("valid_E2") else "GAP"),
        ("public L3 geometry", "DIAG" if not concentration.empty else "MISS"),
        ("own child fills", "PASS" if gate_any("own_fills") else "GAP"),
        ("parent IDs", "PASS" if gate_any("parent_ids") else "GAP"),
        ("terminal horizons", "PASS" if gate_any("terminal_horizons") else "GAP"),
        ("assignment Z", "PASS" if gate_any("assignment_variation") else "GAP"),
        ("E3 residual CIV", "PASS" if gate_any("valid_E3") else "GAP"),
    ]
    color_for = {"PASS": COLORS["blue"], "DIAG": COLORS["gray"], "MISS": COLORS["orange"], "GAP": COLORS["orange"]}
    y0 = 0.93
    for idx, (label, status) in enumerate(claim_rows):
        yy = y0 - idx * 0.086
        ax.text(0.03, yy, label, transform=ax.transAxes, ha="left", va="center", fontsize=7.4, color=COLORS["black"])
        ax.add_patch(
            Rectangle(
                (0.58, yy - 0.035),
                0.31,
                0.07,
                transform=ax.transAxes,
                facecolor=color_for[status],
                edgecolor="white",
                linewidth=0.8,
            )
        )
        ax.text(0.735, yy, status, transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color="white")
    ax.text(
        0.03,
        0.018,
        f"Visible support: {100.0 * float(exact['support_rate']):.0f}%; "
        f"exact RMSE: {float(exact['rmse_ticks']):.2e}; "
        f"rank-1 residual: {rank1_text}.\n"
        "Public L2/L3 stays E2; E3 needs own-execution assignment and terminal outcomes.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color=COLORS["gray"],
    )

    caption = "External L2/L3 data support the displayed-mechanics data contract and book-geometry checks, expose stale-book/proxy/hidden boundaries, and show mechanics-only schedule-ranking consequences; they do not establish terminal propagation."
    return _save(
        fig,
        fig_id=fig_id,
        title="External L2/L3 mechanics boundary",
        claim=caption,
        design_ids=["external_lob_expanded_audit"],
        source_artifacts=[
            "artifacts/external_lob/advanced_l2l3/lobster_staleness_diagnostics.csv",
            "artifacts/external_lob/advanced_l2l3/lobster_proxy_model_diagnostics.csv",
            "artifacts/external_lob/advanced_l2l3/advanced_l2l3_summary.json",
            "artifacts/external_lob/advanced_l2l3/lobster_walk_low_rank_errors.csv",
            "artifacts/external_lob/followup_l2l3/lobster_schedule_ranking_summary.csv",
            "artifacts/external_lob/followup_l2l3/lobster_schedule_decision_summary.csv",
            "artifacts/external_lob/followup_l2l3/coinbase_l3_concentration_summary.csv",
            "artifacts/l2l3_validation/data_contract_gate.csv",
        ],
        source_data=source,
        evidence_type="external_l2_l3_mechanics_audit_only",
        caption=caption,
        extra_metadata={"key_results": summary.get("key_results", {})},
    )


def figure_qpscm_policy_evaluation_stack() -> dict[str, Any]:
    fig_id = "fig_qpscm_policy_evaluation_stack"
    values = read_csv("artifacts/policy_evaluation/policy_value_decomposition.csv")
    diagnostics = read_csv("artifacts/policy_evaluation/policy_regression_diagnostics.csv")
    perturb = read_csv("artifacts/policy_evaluation/policy_perturbation_iv_diagnostics.csv")
    replay = read_csv("artifacts/policy_evaluation/public_l2l3_policy_replay_summary.csv")
    summary = read_json("artifacts/policy_evaluation/summary.json")

    all_values = values[values["regime"].eq("all")].copy()
    policy_order = [
        "random_policy",
        "twap",
        "raw_terminal_rl",
        "qpscm_shaped_rl",
        "proxy_shaped_ablation",
        "oracle_branch_search",
    ]
    all_values = all_values[all_values["policy"].isin(policy_order)].copy()
    all_values["policy"] = pd.Categorical(all_values["policy"], categories=policy_order, ordered=True)
    all_values = all_values.sort_values("policy")

    qpscm_diag = diagnostics[diagnostics["policy"].eq("qpscm_shaped_rl")].copy()
    qpscm_diag["gap"] = qpscm_diag["estimate"] - qpscm_diag["target"]
    qpscm_diag["label"] = qpscm_diag["estimator"].replace(
        {
            "raw_terminal_ols": "raw OLS",
            "residual_ols": "resid. OLS",
            "residual_ols_admissible_controls": "resid. + ctrls",
            "descendant_control_negative": "desc. ctrl",
        }
    )

    perturb_plot = perturb[
        perturb["estimator"].isin(
            [
                "raw_terminal_iv",
                "residual_iv",
                "residual_civ_admissible_controls",
                "weak_instrument_gate",
            ]
        )
    ].copy()
    perturb_plot["gap"] = perturb_plot["estimate"] - perturb_plot["target"]
    perturb_plot["label"] = perturb_plot["estimator"].replace(
        {
            "raw_terminal_iv": "raw IV",
            "residual_iv": "residual IV",
            "residual_civ_admissible_controls": "residual CIV",
            "weak_instrument_gate": "weak gate",
        }
    )

    source = write_source(
        fig_id,
        pd.concat(
            [
                all_values.assign(panel="policy_value_decomposition"),
                qpscm_diag.assign(panel="logged_policy_regressions"),
                perturb_plot.assign(panel="local_randomized_perturbation"),
                replay.assign(panel="public_l2l3_policy_replay"),
            ],
            ignore_index=True,
            sort=False,
        ),
    )

    fig = plt.figure(figsize=(7.8, 6.15), constrained_layout=False)
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.24, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.085,
        right=0.99,
        top=0.95,
        bottom=0.12,
        wspace=0.42,
        hspace=0.58,
    )

    short = {
        "random_policy": "random",
        "twap": "TWAP",
        "liquidity_pov": "POV",
        "raw_terminal_rl": "confounded\nraw fit",
        "qpscm_shaped_rl": "QP-SCM fit",
        "proxy_shaped_ablation": "proxy fit",
        "oracle_branch_search": "oracle",
        "scalar_proxy_policy": "scalar",
        "collapsed_parent": "collapsed",
    }

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(all_values))
    bottom = np.zeros(len(all_values), dtype=float)
    components = [
        ("F_seq", r"$F_{seq}$", COLORS["blue"]),
        ("residual_propagation", "residual", COLORS["orange"]),
        ("risk_penalty", "risk", COLORS["green"]),
    ]
    for col, label, color in components:
        vals = all_values[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, color=color, label=label, width=0.70)
        bottom += vals
    ax.errorbar(
        x,
        all_values["total_cost"],
        yerr=[
            all_values["total_cost"] - all_values["total_cost_ci_low"],
            all_values["total_cost_ci_high"] - all_values["total_cost"],
        ],
        fmt="none",
        ecolor=COLORS["black"],
        elinewidth=0.8,
        capsize=2,
        zorder=4,
    )
    ax.set_xticks(x, [short[str(p)] for p in all_values["policy"]], rotation=25, ha="right")
    ax.set_ylabel("held-out cost (ticks)")
    ax.set_title("A. random vs fitted policy vs oracle", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=7.2)
    ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.55)

    ax = fig.add_subplot(gs[0, 1])
    plot = qpscm_diag.copy()
    plot["plot_gap"] = plot["gap"].clip(-1.05, 0.65)
    y = np.arange(len(plot))[::-1]
    colors = [COLORS["red"] if not bool(v) else COLORS["blue"] for v in plot["valid_graph"]]
    ax.axvline(0, color=COLORS["black"], lw=0.9)
    ax.text(0.03, 0.95, "target", transform=ax.get_xaxis_transform(), fontsize=7.0, color=COLORS["black"], va="top")
    for i, (_, row) in enumerate(plot.iterrows()):
        color = colors[i]
        yi = y[i]
        gap = float(row["gap"])
        plot_gap = float(row["plot_gap"])
        clipped = abs(gap - plot_gap) > 1e-12
        marker = "<" if clipped and gap < plot_gap else (">" if clipped else ("s" if not bool(row["valid_graph"]) else "o"))
        ax.scatter(plot_gap, yi, color=color, s=30, marker=marker, zorder=3)
        if clipped:
            ha = "left" if gap < plot_gap else "right"
            dx = 0.035 if gap < plot_gap else -0.035
            ax.text(plot_gap + dx, yi, f"{gap:+.2f}", ha=ha, va="center", fontsize=6.4, color=color)
        elif pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            lo = max(float(row["ci_low"] - row["target"]), -1.05)
            hi = min(float(row["ci_high"] - row["target"]), 0.65)
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.0)
    ax.set_yticks(y, plot["label"])
    ax.set_xlim(-1.10, 0.70)
    ax.set_xlabel("estimate minus residual target")
    ax.set_title("B. logged actions are endogenous", fontweight="bold")
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.58)

    ax = fig.add_subplot(gs[1, 0])
    plot = perturb_plot.copy()
    plot["plot_estimate"] = plot["estimate"].clip(-0.05, 1.05)
    y = np.arange(len(plot))[::-1]
    target_value = float(plot["target"].dropna().iloc[0])
    ax.axvline(target_value, color=COLORS["black"], lw=1.0)
    ax.text(target_value + 0.012, 0.95, "target", transform=ax.get_xaxis_transform(), fontsize=7.0, color=COLORS["black"], va="top")
    for i, (_, row) in enumerate(plot.iterrows()):
        weak = bool(row["weak_iv_flag"])
        raw = row["estimator"] == "raw_terminal_iv"
        color = COLORS["gray"] if weak else (COLORS["orange"] if raw else COLORS["blue"])
        marker = "D" if weak else ("s" if raw else "o")
        yi = y[i]
        estimate = float(row["estimate"])
        plot_estimate = float(row["plot_estimate"])
        clipped = abs(estimate - plot_estimate) > 1e-12
        marker_use = ">" if clipped and estimate > plot_estimate else ("<" if clipped else marker)
        ax.scatter(plot_estimate, yi, color=color, marker=marker_use, s=34, zorder=3)
        if clipped:
            ha = "right" if estimate > plot_estimate else "left"
            dx = -0.028 if estimate > plot_estimate else 0.028
            ax.text(plot_estimate + dx, yi, f"{estimate:+.2f}", ha=ha, va="center", fontsize=6.4, color=color)
        elif pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
            lo = max(float(row["ci_low"]), -0.05)
            hi = min(float(row["ci_high"]), 1.05)
            ax.plot([lo, hi], [yi, yi], color=color, lw=1.0)
    fs = float(perturb_plot["first_stage_F"].dropna().iloc[0])
    ax.text(0.03, 0.93, f"local first-stage F={fs:.1f}", transform=ax.transAxes, fontsize=7.5, color=COLORS["gray"], va="top")
    ax.set_yticks(y, plot["label"])
    ax.set_xlim(-0.05, 1.08)
    ax.set_xlabel("local randomized slope")
    ax.set_title("C. perturb around fitted policy", fontweight="bold")
    ax.grid(axis="x", color=COLORS["light_gray"], alpha=0.58)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor("#F7F7F7")
    show = replay[replay["policy"].isin(["twap", "qpscm_shaped_rl", "scalar_proxy_policy", "collapsed_parent"])].copy()
    order = ["twap", "qpscm_shaped_rl", "scalar_proxy_policy", "collapsed_parent"]
    show["policy"] = pd.Categorical(show["policy"], categories=order, ordered=True)
    show = show.sort_values("policy")
    x = np.arange(len(show))
    width = 0.36
    ax.bar(x - width / 2, show["median_exact_displayed_cost_ticks"], width=width, color=COLORS["blue"], label="exact")
    ax.bar(x + width / 2, show["median_scalar_proxy_cost_ticks"], width=width, color=COLORS["purple"], label="scalar")
    ax.set_xticks(x, [short[str(p)] for p in show["policy"]], rotation=25, ha="right")
    ax.set_ylabel("median displayed cost (ticks)")
    ax.set_title("D. E2 mechanics-only replay\n(no terminal returns)", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.2)
    ax.grid(axis="y", color=COLORS["light_gray"], alpha=0.55)
    if not show.empty:
        rank = float(show["rank_disagreement_rate"].dropna().iloc[0])
        regret = float(show["p90_scalar_selection_regret_ticks"].dropna().iloc[0])
        ax.text(
            0.03,
            0.94,
            f"rank disagree {rank:.1%}\np90 regret {regret:.1f} ticks",
            transform=ax.transAxes,
            va="top",
            fontsize=7.4,
            color=COLORS["gray"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        )

    caption = "Simulator-local random-policy, confounded raw-terminal fitted-policy, QP-SCM fitted-policy, and oracle benchmark evaluation shows that QP-SCM decomposition changes reward attribution and policy diagnostics; public L2/L3 replay is mechanics-only and does not validate real terminal propagation."
    return _save(
        fig,
        fig_id=fig_id,
        title="QP-SCM policy evaluation stack",
        claim=caption,
        design_ids=["qpscm_policy_evaluation", "external_lob_expanded_audit"],
        source_artifacts=[
            "artifacts/policy_evaluation/policy_value_decomposition.csv",
            "artifacts/policy_evaluation/policy_regression_diagnostics.csv",
            "artifacts/policy_evaluation/policy_perturbation_iv_diagnostics.csv",
            "artifacts/policy_evaluation/public_l2l3_policy_replay_summary.csv",
            "artifacts/policy_evaluation/summary.json",
        ],
        source_data=source,
        evidence_type="simulator_local_policy_evaluation_and_public_mechanics_replay",
        caption=caption,
        extra_metadata={"key_results": summary.get("key_results", {}), "checkpoint_search": summary.get("checkpoint_search", {})},
    )


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def write_table(path: Path, columns: list[tuple[str, str]], rows: list[dict[str, str]], widths: list[str]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    spec = "".join([f"L{{{w}}}" for w in widths])
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    lines.append(" & ".join(header for _, header in columns) + r" \\")
    lines.append("\\midrule")
    for row in rows:
        vals = [row[key] for key, _ in columns]
        lines.append(" & ".join(vals) + r" \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_claim_to_evidence_matrix() -> None:
    rows_raw = [
        ("Exact displayed mechanics", "LOBSTER / cloned walk", "cloned + external L2/L3", "zero/near-zero walk error", "visible support only"),
        ("Scalar proxies are not exact mechanics", "rank-k / normalizer diagnostics", "cloned oracle", "nonzero rank/proxy error", "support-local"),
        ("Proxy mechanics can bias residual IV", "proxy-function bias sweep", "generated overlay", "bias changes sign with error correlation", "generated, not external"),
        ("Residual-CIV works under valid graph", "generated CIV / MC500", "generated overlay", "target coverage/bias", "imposed DGP"),
        ("Native schedules recover schedule target", "native integrated robustness", "native event-engine", r"IV $\approx$ weighted branch target", "simulator-local"),
        ("External validation missing", "external gate", "gate", "no assignment variation", "future work"),
        ("Fitted policy uses decomposition", "policy evaluation stack", "simulator-local", "reward/value decomposes into mechanics/residual/risk", "not real terminal validation"),
    ]
    rows = [
        {
            "claim": tex_escape(a),
            "artifact": tex_escape(b),
            "type": tex_escape(c),
            "criterion": d if "$" in d else tex_escape(d),
            "limitation": tex_escape(e),
        }
        for a, b, c, d, e in rows_raw
    ]
    write_table(
        TABLE_DIR / "table_claim_to_evidence_matrix.tex",
        [("claim", "Claim"), ("artifact", "Main artifact"), ("type", "Evidence type"), ("criterion", "Pass criterion"), ("limitation", "Main limitation")],
        rows,
        ["0.17\\linewidth", "0.19\\linewidth", "0.15\\linewidth", "0.18\\linewidth", "0.15\\linewidth"],
    )


def write_gate_summary_table() -> None:
    mark = {"pass": r"\(\checkmark\)", "diag": r"\(\triangle\)", "miss": r"\(\times\)", "na": "n/a"}
    rows_raw = [
        ("Oracle decomposition atlas", "pass", "pass", "na", "na", "pass", "miss"),
        ("Proxy-error covariance", "pass", "pass", "pass", "pass", "pass", "diag"),
        ("Schedule-path decomposition", "pass", "pass", "diag", "pass", "pass", "miss"),
        ("Residual-CIV validity frontier", "pass", "pass", "diag", "pass", "pass", "miss"),
        ("Native schedule target heatmap", "pass", "pass", "pass", "pass", "pass", "miss"),
        ("External L2/L3 boundary", "pass", "pass", "na", "na", "na", "miss"),
        ("Policy evaluation stack", "pass", "pass", "diag", "diag", "pass", "miss"),
    ]
    rows = [
        {
            "figure": tex_escape(fig),
            "support": mark[support],
            "timestamp": mark[timestamp],
            "first_stage": mark[first_stage],
            "no_desc": mark[no_desc],
            "block": mark[block],
            "external": mark[external],
        }
        for fig, support, timestamp, first_stage, no_desc, block, external in rows_raw
    ]
    write_table(
        TABLE_DIR / "table_gate_summary.tex",
        [
            ("figure", "Figure"),
            ("support", "Support"),
            ("timestamp", "Time"),
            ("first_stage", "First stage"),
            ("no_desc", "No-desc."),
            ("block", "Block inf."),
            ("external", "Ext. assign."),
        ],
        rows,
        ["0.22\\linewidth", "0.08\\linewidth", "0.09\\linewidth", "0.10\\linewidth", "0.09\\linewidth", "0.09\\linewidth", "0.11\\linewidth"],
    )


def main() -> None:
    setup_style()
    entries = [
        figure_oracle_decomposition_atlas(),
        figure_parent_schedule_bundle(),
        figure_civ_validity_frontier(),
    ]
    print(f"Built {len(entries)} public paper evidence figures")


if __name__ == "__main__":
    main()
