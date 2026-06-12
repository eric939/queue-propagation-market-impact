#!/usr/bin/env python3
"""Generate logged Section 3 policy panels from the cloned one-shot ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "data"
    / "market_impact_study"
    / "strategic_groundtruth"
    / "extended_intervention_ladder"
    / "latest"
    / "data"
    / "section3_logged_policy_panel.csv"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_CSV.parents[1] / "metadata" / "section3_logged_policy_summary.json"


POLICY_ORDER = ["randomized", "signal_confounded", "rl_style"]
POLICY_LABELS = {
    "randomized": "Randomized policy",
    "signal_confounded": "Signal-confounded policy",
    "rl_style": "RL-style state policy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-csv", type=Path, default=DEFAULT_BRANCH_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--liquidity-scale", type=float, default=1.5)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--ladder", default=",".join(str(x) for x in DEFAULT_LADDER))
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


def load_supported_branches(args: argparse.Namespace, ladder: tuple[float, ...]) -> pd.DataFrame:
    df = pd.read_csv(args.branch_csv)
    for col in [
        "liquidity_scale",
        "horizon",
        "q_economic_intended",
        "impact_by_horizon",
        "imbalance",
        "bid_depth_total",
        "ask_depth_total",
        "recent_volume_lags",
    ]:
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
        raise ValueError("No supported branches remain after Section 3 filters.")
    expected = set(ladder)
    incomplete = [
        clone_id
        for clone_id, group in rows.groupby("clone_id", sort=False)
        if set(float(q) for q in group["q"].unique()) != expected
    ]
    if incomplete:
        raise ValueError(f"{len(incomplete)} clone states do not have the full requested ladder.")
    return rows


def latent_sign(row: pd.Series) -> int:
    direction = str(row["latent_direction"]).lower()
    if direction == "buy":
        return 1
    if direction == "sell":
        return -1
    raise ValueError(f"Unsupported latent_direction={row['latent_direction']!r}")


def depth_imbalance(row: pd.Series) -> float:
    bid = float(row["bid_depth_total"])
    ask = float(row["ask_depth_total"])
    denom = bid + ask
    return 0.0 if denom <= 1e-12 else float((bid - ask) / denom)


def recent_volume(row: pd.Series) -> float:
    value = float(row.get("recent_volume_lags", 0.0))
    return value if np.isfinite(value) else 0.0


def score_signal_confounded(row: pd.Series) -> float:
    return (
        0.65 * abs(float(row["imbalance"]))
        + 0.25 * abs(depth_imbalance(row))
        + 0.03 * recent_volume(row)
    )


def score_rl_style(row: pd.Series) -> float:
    sign = latent_sign(row)
    pressure = sign * float(row["imbalance"]) + 0.4 * sign * depth_imbalance(row)
    return float(1.0 / (1.0 + np.exp(-8.0 * pressure)) + 0.03 * recent_volume(row))


def quantile_magnitude(score: float, thresholds: np.ndarray) -> int:
    magnitudes = [10, 30, 60, 100]
    return magnitudes[int(np.searchsorted(thresholds, score, side="right"))]


def select_branch(group: pd.DataFrame, q_value: float) -> pd.Series:
    match = group.loc[np.isclose(group["q"].to_numpy(float), float(q_value))]
    if match.empty:
        raise ValueError(f"Missing q={q_value:g} for clone {group['clone_id'].iloc[0]}")
    return match.iloc[0]


def randomized_assignments(clone_ids: list[str], ladder: tuple[float, ...], seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ladder_arr = np.asarray(ladder, dtype=float)
    repeats, remainder = divmod(len(clone_ids), len(ladder_arr))
    assignments = np.tile(ladder_arr, repeats)
    if remainder:
        assignments = np.concatenate([assignments, rng.choice(ladder_arr, size=remainder, replace=False)])
    rng.shuffle(assignments)
    return {clone_id: float(q) for clone_id, q in zip(clone_ids, assignments)}


def policy_rows(rows: pd.DataFrame, ladder: tuple[float, ...], seed: int) -> pd.DataFrame:
    groups = [(clone_id, group.copy()) for clone_id, group in rows.groupby("clone_id", sort=False)]
    clone_ids = [clone_id for clone_id, _ in groups]
    random_q = randomized_assignments(clone_ids, ladder, seed)

    state_rows = [group.iloc[0] for _, group in groups]
    signal_scores = np.asarray([score_signal_confounded(row) for row in state_rows], dtype=float)
    rl_scores = np.asarray([score_rl_style(row) for row in state_rows], dtype=float)
    signal_thresholds = np.quantile(signal_scores, [0.25, 0.50, 0.75])
    rl_thresholds = np.quantile(rl_scores, [0.25, 0.50, 0.75])

    out: list[dict[str, Any]] = []
    for idx, (clone_id, group) in enumerate(groups):
        state = group.iloc[0]
        sign = latent_sign(state)
        assignments = {
            "randomized": random_q[clone_id],
            "signal_confounded": float(sign * quantile_magnitude(signal_scores[idx], signal_thresholds)),
            "rl_style": float(sign * quantile_magnitude(rl_scores[idx], rl_thresholds)),
        }
        scores = {
            "randomized": np.nan,
            "signal_confounded": float(signal_scores[idx]),
            "rl_style": float(rl_scores[idx]),
        }
        for policy_id in POLICY_ORDER:
            branch = select_branch(group, assignments[policy_id])
            out.append(
                {
                    "policy_id": policy_id,
                    "policy_label": POLICY_LABELS[policy_id],
                    "policy_seed": seed,
                    "clone_id": clone_id,
                    "seed": int(branch["seed"]),
                    "episode_id": int(branch["episode_id"]),
                    "latent_direction": branch["latent_direction"],
                    "latent_sign": sign,
                    "q": float(branch["q"]),
                    "impact_by_horizon": float(branch["impact_by_horizon"]),
                    "f_walk": float(branch["f_walk"]),
                    "propagation_residual_by_horizon": float(branch["propagation_residual_by_horizon"]),
                    "policy_score": scores[policy_id],
                    "imbalance": float(branch["imbalance"]),
                    "depth_imbalance": depth_imbalance(branch),
                    "recent_volume_lags": recent_volume(branch),
                    "branch_id": branch["branch_id"],
                }
            )
    return pd.DataFrame(out)


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


def main() -> None:
    args = parse_args()
    ladder = parse_ladder(args.ladder)
    branches = load_supported_branches(args, ladder)
    panel = policy_rows(branches, ladder, args.seed)

    oracle_curve = branches.groupby("q", observed=True)["impact_by_horizon"].mean().reindex(ladder)
    oracle_fit = fit_ols(oracle_curve.index.to_numpy(float), oracle_curve.to_numpy(float))
    policy_fits = {
        policy_id: fit_ols(
            group["q"].to_numpy(float),
            group["impact_by_horizon"].to_numpy(float),
        )
        for policy_id, group in panel.groupby("policy_id", sort=False)
    }
    counts = {
        policy_id: {str(float(q)): int(n) for q, n in group.groupby("q", observed=True).size().sort_index().items()}
        for policy_id, group in panel.groupby("policy_id", sort=False)
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.output_csv, index=False)
    summary = {
        "command": "python scripts/experiments/run_section3_policy_assignment_experiment.py",
        "input_paths": [rel(args.branch_csv)],
        "output_csv": rel(args.output_csv),
        "liquidity_scale": args.liquidity_scale,
        "horizon": args.horizon,
        "ladder": list(ladder),
        "policy_seed": args.seed,
        "n_clones": int(branches["clone_id"].nunique()),
        "n_policy_rows": int(len(panel)),
        "oracle_fit": oracle_fit,
        "policy_fits": policy_fits,
        "policy_assignment_counts": counts,
        "policy_definitions": {
            "randomized": "complete randomized one-branch-per-clone assignment over the plotted ladder",
            "signal_confounded": "latent-direction-aligned sign and magnitude increasing in absolute state/signal score",
            "rl_style": "latent-direction-aligned nonlinear state policy score, used as an RL-style logged policy negative control",
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_csv": rel(args.output_csv), "summary_json": rel(args.summary_json), "fits": policy_fits}, indent=2))


if __name__ == "__main__":
    main()
