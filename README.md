# Queue-Propagation Market Impact: Reproduction Code

This repository contains the simulator code, experiment scripts, compact source data, and figure builders needed to reproduce the paper figures for *Queue-Propagation Market Impact*.

## Contents

- `simulation/`: core baseline simulator code used by the experiments.
- `config/`, `limit_order_book/`, and `initial_shape/`: runtime dependencies for the baseline simulator.
- `simulator_upgrades/`: the simulator modification layer.
- `scripts/experiments/`: figure-specific experiment and plotting scripts used by the reproduction entry point.
- `plots/`: plotting helpers and the selected evidence-figure builders.
- `data/` and `artifacts/`: compact figure-source CSV/JSON inputs for the active paper figures.
- `figures/main/`: paper figure PDFs generated from those inputs.

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

It prints the rebuilt figure paths and SHA-256 hashes.

## Reproducibility Notes

The checked-in CSV/JSON files are the compact source artifacts used by the plotting scripts.  Long-running simulator grids can be regenerated from the simulator and experiment code.  The default workflow uses the compact source artifacts so the paper figures can be reproduced quickly on a clean machine.
