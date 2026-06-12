# Queue-Propagation Market Impact: Public Reproduction Bundle

This public branch is intentionally limited to the self-contained code and data needed to reproduce the active paper figures.  It excludes the manuscript source, LaTeX build products, local drafts, notebooks, submission packaging, and private development artifacts.

## Contents

- `simulation/`: core baseline simulator code used by the experiments.
- `config/`, `limit_order_book/`, and `initial_shape/`: runtime dependencies for the baseline simulator.
- `simulator_upgrades/`: the simulator modification layer.
- `scripts/experiments/`: figure-specific experiment and plotting scripts used by the reproduction entry point.
- `plots/`: plotting helpers and the selected evidence-figure builders.
- `data/` and `artifacts/`: compact figure-source CSV/JSON inputs for the active paper figures.
- `figures/main/`: paper figure PDFs and PNGs generated from those inputs.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/reproduce_paper_figures.py
python3 scripts/check_public_bundle.py
```

The reproduction command rebuilds:

- `fig_section3_policy_regression_bias`
- `fig_oracle_decomposition_atlas`
- `fig_parent_schedule_bundle`
- `fig_functional_civ_curve_validation`
- `fig_civ_validity_frontier`

It also writes `artifacts/reproduction/figure_rebuild_manifest.json` with output paths and SHA-256 hashes.

## Scope

The checked-in CSV/JSON files are the compact source artifacts used by the plotting scripts.  Long-running simulator grids can be regenerated from the simulator and public experiment code, but the default workflow uses these compact source artifacts so the paper figures can be reproduced quickly on a clean machine.
